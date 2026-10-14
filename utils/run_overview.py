#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Run overview in TensorBoard: what a run was trained on and with which model
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      RailTwin addition.
#        HPARAMS tab: one row per run (data, inputs, classes, model, training settings) next to its metrics and
#                     per-class IoUs, to compare runs (table, parallel coordinates, scatter plots).
#        TEXT tab:    "Run/Overview" with the classes and their share of the training points, the inputs, the
#                     architecture and the full configuration.
#      Start TensorBoard on the folder that holds all runs (e.g. the work dir) to see them side by side.
#

import inspect
from pathlib import Path

import numpy as np
from torch.utils.tensorboard.summary import hparams

# Scalar tags written during validation, shown as metric columns in the HPARAMS tab (plus Test/IoU/<class>)
METRICS = ["Test/mIoU", "Test/mIoU_best", "Test/IoU_worst", "Test/Accuracy", "Test/Loss"]


def _text(value):
    """Lists, dicts, None as short strings (HParams only takes numbers, strings and bools)."""
    if isinstance(value, (bool, int, float, str)):
        return value
    if value is None:
        return "none"
    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def class_table(dataset):
    """[(label value, name, ignored, points in the training clouds)] (points: None if the dataset has no labels)."""
    counts = None
    input_labels = getattr(dataset, "input_labels", None)
    if input_labels is not None and len(input_labels):
        # Counted per cloud: no copy of all labels at once
        counts = {}
        for labels in input_labels:
            values, n = np.unique(np.asarray(labels), return_counts=True)
            for value, count in zip(values.tolist(), n.tolist()):
                counts[int(value)] = counts.get(int(value), 0) + count
    return [(int(value), dataset.label_to_names[value], value in dataset.ignored_labels,
             None if counts is None else counts.get(int(value), 0))
            for value in dataset.label_values]


def run_hparams(config, net, dataset, start="", classes=None):
    """The settings that make runs differ, as a flat dict. classes: class_table(dataset), if already computed."""
    architecture = list(config.architecture)
    classes = class_table(dataset) if classes is None else classes
    trained_classes = [name for _, name, ignored, _ in classes if not ignored]
    return {
        # Data
        "data/dataset": _text(config.dataset),
        "data/path": _text(getattr(dataset, "path", "")),
        "data/train_clouds": len(getattr(dataset, "files", []) or []),
        "data/first_subsampling_dl": float(config.first_subsampling_dl),
        "data/in_radius": float(config.in_radius),
        # Inputs and classes
        "inputs/features": _text(config.input_features if config.input_features else "legacy"),
        "inputs/in_features_dim": int(config.in_features_dim),
        "classes/num_trained": len(trained_classes),
        "classes/trained": ", ".join(trained_classes),
        "classes/ignored": _text([dataset.label_to_names[v] for v in dataset.ignored_labels]),
        # Model
        "model/layers": len(architecture),
        "model/deformable": any("deformable" in block for block in architecture),
        "model/first_features_dim": int(config.first_features_dim),
        "model/num_kernel_points": int(config.num_kernel_points),
        "model/conv_radius": float(config.conv_radius),
        "model/parameters": int(sum(p.numel() for p in net.parameters())),
        # Training
        "train/start": start or "scratch",
        "train/batch_num": int(config.batch_num),
        "train/learning_rate": float(config.learning_rate),
        "train/max_epoch": int(config.max_epoch),
        "train/epoch_steps": int(config.epoch_steps),
        "train/loss": _text(config.segmentation_loss),
        "train/balance_classes": _text(config.balance_classes),
        "train/label_smoothing": float(config.label_smoothing),
        "train/unfreeze_schedule": _text(config.unfreeze_schedule),
        # Augmentation
        "augment/rotation": _text(config.augment_rotation),
        "augment/als": float(config.augment_als),
        "augment/mix3d": float(config.augment_mix3d),
        "augment/noise": float(config.augment_noise),
    }


def overview_markdown(config, net, dataset, hparam_dict, classes=None):
    """Markdown for the TEXT tab. classes: class_table(dataset), if already computed."""
    lines = ["### Classes", "", "| label | name | trained | training points | share |", "|---|---|---|---|---|"]
    table = class_table(dataset) if classes is None else classes
    total = sum(n for _, _, ignored, n in table if n is not None and not ignored) or 1
    for value, name, ignored, n in table:
        points = "" if n is None else f"{n:,}"
        share = "" if n is None or ignored else f"{100 * n / total:.2f} %"
        lines.append(f"| {value} | {name} | {'no' if ignored else 'yes'} | {points} | {share} |")

    lines += ["", "### Settings", "", "| setting | value |", "|---|---|"]
    lines += [f"| {key} | {value} |" for key, value in hparam_dict.items()]

    lines += ["", "### Architecture", ""]
    lines += [f"{i}. {block}" for i, block in enumerate(config.architecture)]
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    lines += ["", f"Parameters: {sum(p.numel() for p in net.parameters()):,} ({trainable:,} trainable at start)"]

    lines += ["", "### Full configuration", "", "| name | value |", "|---|---|"]
    for name in sorted(dir(config)):
        # Plain settings only: no methods and no properties (some need settings that a run may not have)
        if name.startswith("_") or isinstance(inspect.getattr_static(config, name), property):
            continue
        value = getattr(config, name)
        if callable(value):
            continue
        lines.append(f"| {name} | {_text(value)} |")
    return "\n".join(lines)


def log_run_overview(writer, config, net, dataset, start=""):
    """Write the HParams entry and the text overview of a run into its TensorBoard run folder."""
    classes = class_table(dataset)
    hparam_dict = run_hparams(config, net, dataset, start, classes)
    # Metric columns: the overall metrics and the IoU of every trained class (all in percent)
    metrics = METRICS + [f"Test/IoU/{name}" for _, name, ignored, _ in classes if not ignored]
    for summary in hparams(hparam_dict, {tag: 0.0 for tag in metrics}):
        writer.file_writer.add_summary(summary)
    writer.add_text("Run/Overview", overview_markdown(config, net, dataset, hparam_dict, classes), 0)
    writer.flush()
    return hparam_dict


def describe_start(chkp_path, finetune):
    """'scratch', 'resume: <run>' or 'finetune: <run>' (run = folder of the checkpoint, above 'checkpoints')."""
    if chkp_path is None:
        return "scratch"
    folder = Path(chkp_path).resolve().parent
    run = (folder.parent if folder.name == "checkpoints" else folder).name
    return f"{'finetune' if finetune else 'resume'}: {run}"
