#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Evaluation of a model on labelled clouds at full resolution (used by evaluate.py)
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      RailTwin addition.
#        confusion()        confusion matrix of the trained classes (points with ignored or unknown labels are left
#                           out), chunked so that 1e9 points need no large temporaries
#        scores()           IoU per class, mIoU, overall accuracy (IoU and accuracy in percent)
#        ground_truth()     labels of a cloud in the classes of the model (label mapping applied if the model has one)
#        write_tensorboard  an "eval_<name>" run next to the training runs: the model's HParams with the evaluation
#                           metrics, so evaluations on the same test set are rows of one HParams table
#

import numpy as np


def confusion(truth, pred, label_values, ignored_labels=(), chunk=50_000_000):
    """
    Confusion matrix C[true, predicted] over the trained classes (label_values without ignored_labels), int64.
    Points whose true label is ignored or not a label value are not counted; their number is returned too.
    :return: (C, class label values in the order of C, number of points left out)
    """
    classes = np.array([v for v in label_values if v not in ignored_labels], dtype=np.int64)
    k = len(classes)
    # label value -> row/column of C (k = left out)
    size = int(max(np.max(label_values), np.max(truth, initial=0), np.max(pred, initial=0))) + 1
    index = np.full(size, k, dtype=np.int64)
    index[classes] = np.arange(k)
    C = np.zeros((k + 1, k + 1), dtype=np.int64)
    for start in range(0, len(truth), chunk):
        t = index[np.asarray(truth[start:start + chunk], dtype=np.int64)]
        p = index[np.asarray(pred[start:start + chunk], dtype=np.int64)]
        C += np.bincount(t * (k + 1) + p, minlength=(k + 1) ** 2).reshape(k + 1, k + 1)
    left_out = int(C[k].sum())
    # Predictions outside the trained classes are wrong predictions of the true class: keep them as a column
    # outside the classes (counted as false negatives), drop the row of the left-out points
    return C[:k], classes, left_out


def scores(C, names):
    """IoU per class, mIoU, accuracy (percent) from a confusion matrix C[true, predicted] (k x k or k x k+1)."""
    k = C.shape[0]
    tp = np.diag(C[:, :k]).astype(np.float64)
    true_n = C.sum(axis=1)                 # points of the class (also those predicted outside the classes)
    pred_n = C[:, :k].sum(axis=0)          # points predicted as the class
    union = true_n + pred_n - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, tp / union, np.nan)
    total = C.sum()
    present = true_n > 0
    return {
        "points": int(total),
        "accuracy": float(100 * tp.sum() / total) if total else float("nan"),
        # mean over the classes that occur in the ground truth
        "mIoU": float(100 * np.nanmean(iou[present])) if present.any() else float("nan"),
        "IoU": {name: (None if np.isnan(v) else float(100 * v)) for name, v in zip(names, iou)},
        "class_points": {name: int(n) for name, n in zip(names, true_n)},
    }


def ground_truth(pointcloud, config, label_column=None):
    """
    True labels of a cloud in the classes of the model.
    label_column given: that column, as class ids of the model. Else: with a label mapping in the config, its
    label_column mapped through the mapping; without, the column config.pred_selected (e.g. 'system').
    """
    if label_column:
        return pointcloud[label_column].to_numpy()
    if getattr(config, "label_mapping", None):
        from railtwin_common.labels import LabelChain
        chain = LabelChain.load(config.label_mapping, config.label_level)
        return chain.remap(pointcloud[config.label_column].to_numpy())
    return pointcloud[config.pred_selected].to_numpy()


def report_markdown(name, result):
    """Markdown table of one result (scores())."""
    lines = [f"### {name}", "",
             f"mIoU {result['mIoU']:.1f} %, accuracy {result['accuracy']:.1f} %, {result['points']:,} points", "",
             "| class | IoU [%] | points |", "|---|---|---|"]
    for cls, iou in result["IoU"].items():
        lines.append(f"| {cls} | {'-' if iou is None else f'{iou:.1f}'} | {result['class_points'][cls]:,} |")
    return "\n".join(lines)


def training_hparams(log_dir):
    """HParams of a training run (utils/run_overview.py) from its TensorBoard folder, {} if there are none."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.plugins.hparams import plugin_data_pb2
    events = EventAccumulator(str(log_dir), size_guidance={"tensors": 0})
    events.Reload()
    content = events.PluginTagToContent("hparams").get("_hparams_/session_start_info")
    if content is None:
        return {}
    info = plugin_data_pb2.HParamsPluginData.FromString(content).session_start_info
    result = {}
    for key, value in info.hparams.items():
        kind = value.WhichOneof("kind")
        result[key] = getattr(value, kind) if kind else ""
    return result


def write_tensorboard(log_dir, hparam_dict, total, per_cloud, text):
    """Evaluation as a TensorBoard run: HParams of the model + Eval/* metrics, and the report as text."""
    from pathlib import Path
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.tensorboard.summary import hparams
    # A repeated evaluation replaces the earlier one (one HParams row per model and test set)
    for old in Path(log_dir).glob("events.out.tfevents.*"):
        old.unlink()
    writer = SummaryWriter(str(log_dir))
    metrics = {"Eval/mIoU": total["mIoU"], "Eval/accuracy": total["accuracy"]}
    metrics.update({f"Eval/IoU/{cls}": iou for cls, iou in total["IoU"].items() if iou is not None})
    for name, result in per_cloud.items():
        writer.add_scalar(f"Eval_cloud/{name}/mIoU", result["mIoU"], 0)
    # Session summaries only (see utils/run_overview.py): the columns come from the scalars of all runs
    _, session_start, session_end = hparams(hparam_dict, metrics)
    writer.file_writer.add_summary(session_start)
    writer.file_writer.add_summary(session_end)
    for tag, value in metrics.items():
        writer.add_scalar(tag, value, 0)
    writer.add_text("Eval/Report", text, 0)
    writer.close()
