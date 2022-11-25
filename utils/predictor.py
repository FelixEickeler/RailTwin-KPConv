# 24.06.22----------------------------------------------------------------------------------------------------------------------
#  created by: Felix Eickeler 
#              felix.eickeler@tum.de       
# ----------------------------------------------------------------------------------------------------------------
#
#
import os
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import pandas
import torch
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader

from datasets.RailTwin_priming import S3DISCollate
from datasets.common import grid_subsampling
from datasets.inference_dataset import InferenceDataset, InferenceSampler
from models.architectures import KPFCNN
from railtwin_common.algorithms.voting import KnnVoting
from railtwin_common.common.io.io_options import IOOptions
from railtwin_common.common.io.reader.ply import read_ply
from railtwin_common.common.io.writer.ply import ply_writer
from railtwin_common.common.logger import RailTwinLogger

# from test_railtwin import logger, cfg_path

logger = RailTwinLogger.create()


class Predictor:

    def __init__(self, cfg):
        self.test_probs = None
        self.model = None
        self.config = cfg.copy()
        self.selected_property = f"preds_{self.config.pred_selected}"
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  #:0
        else:
            self.device = torch.device("cpu")

        if hasattr(self.config, "version") and self.config.version >= 2.0:
            logger.info("Generating KPFCNN")
            self.model = KPFCNN(self.config, self.config.labels, self.config.ignored_labels)  #
        else:
            logger.warn("Your KPconf config does not support this testing methodology, please use the orignal Testing")
            raise Exception("Deprecated")

        self.model.to(self.device)
        self.epoch = 0
        checkpoint = torch.load(cfg.chkp_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # rolling temporaries
        folders = list(Path(cfg.saving_path).glob("*_predictions"))
        if len(folders) > 4:
            folders.sort(key=lambda x: os.path.getmtime(x))
            for f in range(len(folders) - 4):
                shutil.rmtree(folders[f])

        self.folder = Path(cfg.saving_path) / f"{time.strftime('%Y-%m-%d')}_predictions"
        self.folder.mkdir(parents=True, exist_ok=True)

        self.test_smooth = 0.95
        self.test_radius_ratio = 0.7
        self.softmax = torch.nn.Softmax(1)
        self.num_votes = 100 #100

    def subsample(self, pointcloud, tree_path, cache_name):
        # subsample
        logger.info("Subsampling points...")
        dl = self.config.first_subsampling_dl
        tmp_set = pointcloud[self.config.features].to_numpy(np.float32)
        labels = pointcloud[self.config.pred_names].to_numpy(np.int32)
        points, features, labels = grid_subsampling(tmp_set[:, :3], tmp_set[:, 3:], labels, sampleDl=dl)

        subsampled = pandas.DataFrame(np.hstack([points, features]), columns=self.config.features).join(
            pandas.DataFrame(labels, columns=self.config.pred_names, dtype=np.int32))

        search_tree = KDTree(points, leaf_size=10)
        with open(tree_path, 'wb') as f:
            pickle.dump(search_tree, f)

        cloud_path = tree_path.parent / f"{cache_name}.ply"
        io_options = IOOptions.do_nothing()
        io_options.binary = True
        ply_writer(cloud_path, subsampled, io_options)
        return cloud_path, subsampled, search_tree, features, labels

    def infer(self, test_loader, dataset, subsampled, inference_path):


        logger.info("Start prediction")
        test_epoch = 0
        last_display = time.time()
        moving_avg = 0
        avg_count = 10
        logger.info(f"Batchsize: {self.config.val_batch_num} \n "
                    f"Batchsize Validation {self.config.val_batch_num} \n "
                    f"Validationsize: {self.config.validation_size}")

        while True:
            for i, batch in enumerate(test_loader):
                start_time = time.time()
                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = self.model(batch, self.config)
                forward_time = time.time()

                # Get probs and labels
                stacked_probs = self.softmax(outputs).cpu().detach().numpy()
                s_points = batch.points[0].cpu().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                in_inds = batch.input_inds.cpu().numpy()
                cloud_inds = batch.cloud_inds.cpu().numpy()
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()

                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    points = s_points[i0:i0 + length]
                    probs = stacked_probs[i0:i0 + length]
                    inds = in_inds[i0:i0 + length]

                    # mask only inner points of sphere
                    if 0 < self.test_radius_ratio < 1:
                        mask = np.sum(points ** 2, axis=1) < (self.test_radius_ratio * self.config.in_radius) ** 2
                        inds = inds[mask]
                        probs = probs[mask]

                    # Update current probs in whole cloud
                    self.test_probs[inds] = self.test_smooth * self.test_probs[inds] + (1 - self.test_smooth) * probs
                    i0 += length

                loop_end_time = time.time()
                loop_time = loop_end_time - start_time
                moving_avg = moving_avg * (avg_count - 1) / avg_count + loop_time / avg_count
                if (loop_end_time - last_display) > 5.0:
                    last_display = loop_end_time
                    message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f})'
                    print(message.format(test_epoch, i,
                                         100 * i / self.config.validation_size,  # percent
                                         (forward_time - start_time) * 1000,
                                         loop_time * 1000,
                                         moving_avg * 1000))

            # Update minimum od potentials
            new_min = torch.min(test_loader.dataset.min_potentials)
            logger.info('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))
            test_epoch += 1

            # Break when reaching number of desired votes
            if new_min > self.num_votes:
                break
            # else:
            #     last_min = new_min

        proj_probs = self.test_probs.copy()

        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
            if label_value in test_loader.dataset.ignored_labels:
                proj_probs = np.insert(proj_probs, l_ind, 0, axis=1)

        prop_names = ["x", "y", "z"] + self.config.pred_names
        extract = subsampled[prop_names].join(pandas.Series(dataset.label_values[np.argmax(proj_probs, axis=1)],
                                                            dtype=np.int32,
                                                            name=self.selected_property))
        io_options = IOOptions.do_nothing()
        io_options.binary = True
        ply_writer(output_path=inference_path, point_cloud=extract, options=io_options)
        print(f"Prediction completed. Intermediate path: {inference_path}")
        return extract

    @torch.no_grad()
    def run_on_cloud(self, pointcloud: pandas.DataFrame, cache_name="tmp"):


        inference_path = self.folder / f"{cache_name}_predictions.ply"
        tree_folder = self.folder  # / "KDTrees"
        tree_path = tree_folder / '{}_KDTree_{:.3f}.pkl'.format(cache_name, self.config.first_subsampling_dl)

        if not inference_path.exists():
            # subsample
            subsampled_path, subsampled, subtree, features, labels = self.subsample(pointcloud=pointcloud, tree_path=tree_path, cache_name=cache_name)

            # Setup Dataset
            tree_folder.mkdir(parents=True, exist_ok=True)
            dataset = InferenceDataset(self.config, cloud_path=subsampled_path, tree_path=tree_path, use_potentials=True)
            dataset.input_features = [features]  # subsampled[self.config.features[3:]]
            dataset.input_labels = [labels]  # subsampled[self.config.pred_names]
            sampler = InferenceSampler(dataset)
            test_loader = DataLoader(dataset,
                                     batch_size=1,
                                     sampler=sampler,
                                     collate_fn=S3DISCollate,
                                     num_workers=self.config.input_threads,
                                     pin_memory=True)

            # Calibrate input Batches
            logger.info("Calibrating input batches...")
            sampler.calibration(dataloader=test_loader, verbose=True, force_redo=True)
            self.test_probs = np.zeros((dataset.input_labels[0].shape[0], self.config.num_classes))

            # Predictions
            logger.info("starting prediction ...")
            prediction = self.infer(test_loader=test_loader, dataset=dataset,
                                    subsampled=subsampled, inference_path=inference_path)
        else:
            logger.info("Predictions found: {inference_path.name}")
            prediction = read_ply(inference_path)
            with open(tree_path, 'rb') as f:
                subtree = pickle.load(f)

        logger.info(f"Big play with {len(prediction)} of points")

        knn_voting_scheme = KnnVoting(prediction, pointcloud, self.selected_property, search_tree=subtree,knn=5)
        logger.info("Now starting apply")
        pointcloud[self.selected_property] = pointcloud.apply(knn_voting_scheme, axis=1, raw=False)
        # print(f"in seconds: {t2 -t1}")
        logger.info("PointCloud ready")
        return pointcloud
