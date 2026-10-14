#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Find the checkpoint of a run (training, fine-tuning and prediction use the same rule)
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      RailTwin addition
#

import os
from pathlib import Path


def find_checkpoint(path):
    """
    Return the checkpoint file itself, or the newest *.tar of a run folder (or its checkpoints/ subfolder).
    The newest is usually current_chkp.tar, which holds the last epoch. None if there is none.
    """
    path = Path(path).expanduser()
    if path.is_file():
        return path.resolve()
    chkp_dir = path / 'checkpoints' if (path / 'checkpoints').is_dir() else path
    chkps = sorted(chkp_dir.glob('*.tar'), key=os.path.getmtime)
    return chkps[-1].resolve() if chkps else None
