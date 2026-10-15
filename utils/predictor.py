# 24.06.22----------------------------------------------------------------------------------------------------------------------
#  created by: Felix Eickeler 
#              felix.eickeler@tum.de       
# ----------------------------------------------------------------------------------------------------------------
#
#
import hashlib
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

from datasets.railtwin import RailTwinCollate
from datasets.common import grid_subsampling
from datasets.inference_dataset import InferenceDataset, InferenceSampler
from models.architectures import KPFCNN
from utils.voting import knn_voting
from railtwin_common.common.io.io_options import IOOptions
from railtwin_common.common.io.reader.ply import read_ply
from railtwin_common.common.io.writer.ply import ply_writer
from railtwin_common.common.logger import RailTwinLogger

# from test_railtwin import logger, cfg_path

logger = RailTwinLogger.create()


class Predictor:

    def __init__(self, cfg, num_votes=100, recalibrate=False):
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
            logger.warning("Your KPconf config does not support this testing methodology, please use the orignal Testing")
            raise Exception("Deprecated")

        self.model.to(self.device)
        self.epoch = 0
        checkpoint = torch.load(cfg.chkp_path, map_location=self.device)
        # Cached predictions are only valid for this checkpoint, subsampling and number of votes
        chkp = Path(cfg.chkp_path)
        self.model_key = hashlib.sha1(f"{chkp.resolve()}|{chkp.stat().st_mtime}|{chkp.stat().st_size}|"
                                      f"{self.config.first_subsampling_dl}|{num_votes}".encode()).hexdigest()[:10]
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
        # Inference stops when every point of the subsampled cloud was predicted about num_votes times
        self.num_votes = num_votes
        # Batch and neighbour calibration, stored per density class (see density_class) and reused for the next
        # clouds; recalibrate=True computes it again
        self.recalibrate = recalibrate
        self.calibration_dir = Path(cfg.saving_path) / "calibration"
        self.calibration_dir.mkdir(parents=True, exist_ok=True)

    def density_class(self, search_tree, samples=200):
        """
        Density class of a subsampled cloud for the calibration key: median number of points in a sphere of
        in_radius around random points, in steps of 2^(1/4) (about 19 %). Clouds of similar density (e.g. the same
        sensor) share a calibration, others (e.g. ALS vs MLS) get their own.
        """
        points = np.asarray(search_tree.data)
        rng = np.random.default_rng(0)
        centres = points[rng.choice(len(points), min(samples, len(points)), replace=False)]
        counts = search_tree.query_radius(centres, r=self.config.in_radius, count_only=True)
        return int(round(4 * np.log2(max(np.median(counts), 1))))

    def stored_density(self, density):
        """
        The density class to calibrate with: a stored one next to it (one class, about 19 %) if there is one, so that
        clouds of one source near a class border share the calibration. Otherwise density itself (new calibration).
        """
        logger.info(f"Density class {density} (median {2 ** (density / 4):.0f} points per sphere)")
        if self.recalibrate:
            return density
        stored = self.calibration_dir / "batch_limits.pkl"
        keys = pickle.load(open(stored, "rb")).keys() if stored.exists() else []
        for candidate in (density, density - 1, density + 1):
            if any(k.endswith(f"_density{candidate}") for k in keys):
                if candidate != density:
                    logger.info(f"Using the stored calibration of density class {candidate}")
                return candidate
        return density

    def subsample(self, pointcloud, local_xyz, cache_name):
        """Grid subsampling of the cloud; local_xyz: float32[N, 3] local coordinates (no copy of the whole cloud)."""
        logger.info("Subsampling points...")
        dl = self.config.first_subsampling_dl
        features = pointcloud[self.config.features[3:]].to_numpy(np.float32)
        label_values = list(self.config.labels.values()) if isinstance(self.config.labels, dict) else [0]
        labels = None
        if all(name in pointcloud for name in self.config.pred_names):
            labels = pointcloud[self.config.pred_names].to_numpy(np.int32)
            if not np.isin(labels, label_values).all():
                # Labels of other classes (e.g. a model trained with a label mapping): not usable
                logger.info("The label columns of the cloud do not match the classes of the model, not used")
                labels = None
        if labels is None:
            # Unlabelled cloud: placeholder labels (a valid label value, not used for the prediction)
            labels = np.full((len(pointcloud), len(self.config.pred_names)), min(label_values), dtype=np.int32)
        points, features, labels = grid_subsampling(local_xyz, features, labels, sampleDl=dl)

        subsampled = pandas.DataFrame(np.hstack([points, features]), columns=self.config.features).join(
            pandas.DataFrame(labels, columns=self.config.pred_names, dtype=np.int32))

        search_tree = KDTree(points, leaf_size=10)

        cloud_path = self.folder / f"{cache_name}.ply"
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
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)

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
        cache_name = f"{cache_name}_{self.model_key}"

        # Work in local coordinates: float32 is not precise enough for georeferenced coordinates, but for local ones.
        # One float32 array of the coordinates is all that is copied of the cloud (1e9 points: 12 GB).
        xyz_names = list(self.config.features[:3])
        offset = np.floor(pointcloud[xyz_names].min().to_numpy(np.float64))
        local_xyz = np.empty((len(pointcloud), 3), dtype=np.float32)
        for i, name in enumerate(xyz_names):
            local_xyz[:, i] = pointcloud[name].to_numpy(np.float64) - offset[i]

        inference_path = self.folder / f"{cache_name}_predictions.ply"

        if not inference_path.exists():
            # subsample
            subsampled_path, subsampled, search_tree, features, labels = self.subsample(pointcloud, local_xyz, cache_name)

            # Setup Dataset
            dataset = InferenceDataset(self.config, cloud_path=subsampled_path, search_tree=search_tree, use_potentials=True)
            dataset.input_features = [features]  # subsampled[self.config.features[3:]]
            dataset.input_labels = [labels]  # subsampled[self.config.pred_names]
            sampler = InferenceSampler(dataset)
            test_loader = DataLoader(dataset,
                                     batch_size=1,
                                     sampler=sampler,
                                     collate_fn=RailTwinCollate,
                                     num_workers=self.config.input_threads,
                                     pin_memory=True)

            # Calibrate input Batches
            logger.info("Calibrating input batches...")
            density = self.stored_density(self.density_class(search_tree))
            sampler.calibration(dataloader=test_loader, verbose=True, force_redo=self.recalibrate,
                                cache_dir=self.calibration_dir, key_suffix=f"_density{density}")
            self.test_probs = np.zeros((dataset.input_labels[0].shape[0], self.config.num_classes))

            # Predictions
            logger.info("starting prediction ...")
            prediction = self.infer(test_loader=test_loader, dataset=dataset,
                                    subsampled=subsampled, inference_path=inference_path)
        else:
            logger.info(f"Predictions found: {inference_path.name}")
            prediction = read_ply(inference_path)

        logger.info(f"Big play with {len(prediction)} of points")

        logger.info("Transferring the labels to the full cloud")
        pointcloud[self.selected_property] = knn_voting(prediction[["x", "y", "z"]].to_numpy(),
                                                        prediction[self.selected_property].to_numpy(), local_xyz, knn=5)
        # print(f"in seconds: {t2 -t1}")
        logger.info("PointCloud ready")
        return pointcloud
