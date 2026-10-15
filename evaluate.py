#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Evaluate a trained model on labelled clouds at full resolution
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      RailTwin addition.
#          python3 evaluate.py -m <run folder | checkpoint> -i <cloud | folder> [-n <test set name>] [--votes 100]
#      Predicts every cloud (utils/predictor.py, the prediction cache is reused), compares with the true labels of the
#      cloud (the classes of the model; a label mapping of the model is applied) and writes
#          <output>/evaluation.json, evaluation.csv   IoU per class, mIoU and accuracy, per cloud and in total
#          <run>/logdir/eval_<name>/                  TensorBoard: HParams row with the Eval/* metrics, report as text
#      Points with ignored labels (e.g. 'ignored') are not counted. IoU and accuracy in percent.
#

import argparse
import csv
import json
import os
import signal
from pathlib import Path

import numpy as np

from railtwin_common.common.io import read
from railtwin_common.common.io.io_options import IOOptions
from railtwin_common.common.io.read_router import scandir_supported
from railtwin_common.common.io.writer.ply import ply_writer
from railtwin_common.common.logger import RailTwinLogger
from test_railtwin import resolve_model, setup_cfg
from utils.evaluation import (confusion, ground_truth, report_markdown, scores, training_hparams,
                              write_tensorboard)
from utils.predictor import Predictor

logger = RailTwinLogger.create()


def run_folder(model_path):
    """Run folder of a checkpoint (the folder above 'checkpoints')."""
    folder = model_path.parent
    return folder.parent if folder.name == "checkpoints" else folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model on labelled clouds')
    parser.add_argument('-m', '--model', type=Path, required=True, help='Run folder, checkpoints folder or .tar')
    parser.add_argument('-i', '--input_path', type=Path, required=True, help='Labelled cloud or folder of clouds')
    parser.add_argument('-o', '--output_path', type=Path, default=None,
                        help='Folder for the report (default: <run>/evaluation/<name>)')
    parser.add_argument('-n', '--name', type=str, default=None, help='Name of the test set (default: input name)')
    parser.add_argument('--label-column', type=str, default=None,
                        help='Column with the true labels as class ids of the model (default: from the model config)')
    parser.add_argument('--save-predictions', action='store_true', help='Also write <cloud>_predicted.ply')
    parser.add_argument('--no-tensorboard', action='store_true')
    parser.add_argument('--votes', type=int, default=100,
                        help='Votes per point (default 100; fewer is faster, e.g. 10-20 for a quick check)')
    parser.add_argument('-gid', '--gpu_id', default="0", type=str, help="GPU id")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    input_path = args.input_path.expanduser()
    if input_path.is_dir():
        files = sorted(Path(p) for p in scandir_supported(input_path) if not Path(p).stem.endswith('_predicted'))
        input_root = input_path
    else:
        files, input_root = [input_path], input_path.parent
    if not files:
        raise FileNotFoundError(f"No clouds in {input_path}")
    name = args.name or input_path.stem

    model_path, cfg_path = resolve_model(args.model)
    run_dir = run_folder(model_path)
    output = args.output_path.expanduser() if args.output_path else run_dir / "evaluation" / name
    output.mkdir(parents=True, exist_ok=True)
    logger.info(f"Model: {model_path}, test set '{name}': {len(files)} clouds, report in {output}")

    cfg = setup_cfg(cfg_path=cfg_path, chkps_path=model_path)
    cfg.val_batch_num = 20
    cfg.batch_num = 10
    cfg.validation_size = 200
    label_values = sorted(cfg.labels.values())
    names_of = {v: k for k, v in cfg.labels.items()}
    ignored = [cfg.labels[n] for n in cfg.ignored_labels]

    predictor = Predictor(cfg, num_votes=args.votes)
    per_cloud, total_C, class_values = {}, None, None
    for file in files:
        relative = file.relative_to(input_root)
        cloud_name = "_".join(relative.with_suffix('').parts)
        pc = read(file)
        truth = ground_truth(pc, cfg, args.label_column)
        pc = predictor.run_on_cloud(pc, cache_name=cloud_name)
        C, class_values, left_out = confusion(truth, pc[predictor.selected_property].to_numpy(), label_values, ignored)
        result = scores(C, [names_of[v] for v in class_values])
        result["left_out_points"] = left_out
        per_cloud[cloud_name] = result
        total_C = C if total_C is None else total_C + C
        logger.info(f"{cloud_name}: mIoU {result['mIoU']:.1f} %, accuracy {result['accuracy']:.1f} % "
                    f"({left_out:,} points with ignored / unknown labels left out)")
        if args.save_predictions:
            options = IOOptions.do_nothing()
            options.binary = True
            target = output / relative.parent / f"{file.stem}_predicted.ply"
            target.parent.mkdir(parents=True, exist_ok=True)
            ply_writer(target, pc, options=options)
        del pc, truth

    total = scores(total_C, [names_of[v] for v in class_values])
    total["left_out_points"] = sum(r["left_out_points"] for r in per_cloud.values())
    text = "\n\n".join([report_markdown(f"{name} (all clouds)", total)] +
                       [report_markdown(n, r) for n, r in per_cloud.items()])
    print(text)

    # Reports
    with open(output / "evaluation.json", "w") as f:
        json.dump({"model": str(model_path), "test_set": name, "votes": args.votes,
                   "total": total, "clouds": per_cloud,
                   "confusion": {"classes": [names_of[v] for v in class_values],
                                 "matrix": total_C.tolist(),
                                 "note": "rows: true class, columns: predicted class (last: outside the classes)"}},
                  f, indent=2)
    with open(output / "evaluation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        classes = list(total["IoU"])
        writer.writerow(["cloud", "points", "accuracy", "mIoU"] + [f"IoU {c}" for c in classes])
        for n, r in [("total", total)] + list(per_cloud.items()):
            writer.writerow([n, r["points"], f"{r['accuracy']:.2f}", f"{r['mIoU']:.2f}"] +
                            ["" if r["IoU"][c] is None else f"{r['IoU'][c]:.2f}" for c in classes])

    # TensorBoard: next to the training runs of the model, with the model's HParams
    if not args.no_tensorboard:
        train_logs = sorted((run_dir / "logdir").glob("*_experiment_*"), key=lambda p: p.stat().st_mtime)
        hparam_dict = training_hparams(train_logs[-1]) if train_logs else {}
        hparam_dict.update({"eval/test_set": name, "eval/model": run_dir.name, "eval/checkpoint": model_path.name,
                            "eval/votes": args.votes})
        write_tensorboard(run_dir / "logdir" / f"eval_{name}", hparam_dict, total, per_cloud, text)
    logger.info(f"Evaluation written to {output}")

    # The DataLoader workers of the predictor can keep the process alive (as in test_railtwin.py)
    os.kill(os.getpid(), signal.SIGINT)
