"""
Tests for the checkpoint lookup (utils/checkpoints.py), used for resuming, fine-tuning and prediction.
"""
import os

from utils.checkpoints import find_checkpoint


def touch(path, mtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'')
    os.utime(path, (mtime, mtime))


def test_newest_checkpoint_of_a_run_folder(tmp_path):
    touch(tmp_path / 'checkpoints' / 'chkp_0050.tar', 100)
    touch(tmp_path / 'checkpoints' / 'current_chkp.tar', 200)
    touch(tmp_path / 'checkpoints' / 'chkp_0100.tar', 150)
    assert find_checkpoint(tmp_path).name == 'current_chkp.tar'
    assert find_checkpoint(tmp_path / 'checkpoints').name == 'current_chkp.tar'


def test_file_is_returned_and_empty_folder_gives_none(tmp_path):
    touch(tmp_path / 'chkp_0050.tar', 100)
    assert find_checkpoint(tmp_path / 'chkp_0050.tar') == (tmp_path / 'chkp_0050.tar').resolve()
    assert find_checkpoint(tmp_path / 'empty') is None
