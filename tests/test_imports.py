import latcorr
from latcorr import analysis, correlators, ground_state, plotting, preprocess, resampling, utils


def test_top_level_import():
    assert latcorr.__version__ == "0.1.0"


def test_core_subpackages_import():
    assert resampling.bootstrap is not None
    assert resampling.jackknife is not None
    assert resampling.bin_data is not None
    assert preprocess is not None
    assert analysis is not None
    assert correlators is not None
    assert plotting is not None
    assert ground_state is not None
    assert utils is not None
    assert correlators.read_pt2_h5 is not None
    assert correlators.read_pt3_h5 is not None
    assert correlators.get_ratio_data is not None
    assert correlators.get_sum_data is not None
    assert correlators.get_fh_data is not None
