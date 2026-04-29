from pathlib import Path

import numpy as np
import pytest

from latcorr.correlators import read_qda_h5


def _qda_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "cache"
        / "comb_qTMDWF"
        / "qTMDWF_CG_1HYP_M140_GSRC_W45_k0_src5_OT5.h5"
    )


def test_read_qda_h5_single_dataset():
    path = _qda_path()
    if not path.exists():
        pytest.skip("DA h5 file not present")

    data = read_qda_h5(path, bT="bT0", bz="bz0")

    assert isinstance(data, np.ndarray)
    assert data.shape == (64, 700)
    assert np.iscomplexobj(data)


def test_read_qda_h5_bt_mapping():
    path = _qda_path()
    if not path.exists():
        pytest.skip("DA h5 file not present")

    bt_data = read_qda_h5(path, bT="bT0")

    assert isinstance(bt_data, dict)
    assert "bz0" in bt_data
    assert bt_data["bz0"].shape == (64, 700)


def test_read_qda_h5_requires_bt_for_bz():
    path = _qda_path()
    if not path.exists():
        pytest.skip("DA h5 file not present")

    with pytest.raises(ValueError):
        read_qda_h5(path, bz="bz0")
