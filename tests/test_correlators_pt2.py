from pathlib import Path

import numpy as np
import pytest

from latcorr.correlators import read_pt2_h5


def _k0_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "cache"
        / "comb_c2pt"
        / "c2pt_merged_src5_1HYP_M140_GSRC_W45_k0.h5"
    )


def test_read_pt2_h5_single_dataset():
    path = _k0_path()
    if not path.exists():
        pytest.skip("k0 h5 file not present")

    data = read_pt2_h5(path, source_sink="SS", gamma="5", momentum="PX0PY0PZ0")

    assert isinstance(data, np.ndarray)
    assert data.shape == (64, 700)
    assert np.iscomplexobj(data)


def test_read_pt2_h5_gamma_mapping():
    path = _k0_path()
    if not path.exists():
        pytest.skip("k0 h5 file not present")

    gamma_data = read_pt2_h5(path, source_sink="SS", gamma="5")

    assert isinstance(gamma_data, dict)
    assert "PX0PY0PZ0" in gamma_data
    assert gamma_data["PX0PY0PZ0"].shape == (64, 700)


def test_read_pt2_h5_requires_gamma_for_momentum():
    path = _k0_path()
    if not path.exists():
        pytest.skip("k0 h5 file not present")

    with pytest.raises(ValueError):
        read_pt2_h5(path, source_sink="SS", momentum="PX0PY0PZ0")
