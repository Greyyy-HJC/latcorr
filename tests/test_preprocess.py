import numpy as np
import pytest

from latcorr.preprocess import (
    average_sources,
    drop_nonfinite_samples,
    merge_configurations,
    normalize_correlator,
    preprocess_correlator,
    preprocess_nucleon_tmdpdf,
    slice_time_extent,
    symmetrize_correlator,
)


def test_drop_nonfinite_samples_removes_bad_rows():
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, np.nan, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )

    out = drop_nonfinite_samples(data, sample_axis=0)

    np.testing.assert_allclose(out, np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]))


def test_slice_time_extent():
    data = np.arange(12).reshape(2, 6)

    out = slice_time_extent(data, tmin=1, tmax=4, time_axis=1)

    np.testing.assert_allclose(out, np.array([[1, 2, 3], [7, 8, 9]]))


def test_symmetrize_correlator_periodic():
    data = np.array([[1.0, 2.0, 3.0, 2.0]])

    out = symmetrize_correlator(data, boundary="periodic", time_axis=1)

    np.testing.assert_allclose(out, np.array([[1.5, 2.0, 2.0, 1.5]]))


def test_symmetrize_correlator_antiperiodic():
    data = np.array([[1.0, 2.0, -3.0, -2.0]])

    out = symmetrize_correlator(data, boundary="anti-periodic", time_axis=1)

    np.testing.assert_allclose(out, np.array([[1.5, 2.0, -2.0, -1.5]]))


def test_normalize_correlator():
    data = np.array([[2.0, 4.0, 8.0], [1.0, 2.0, 4.0]])

    out = normalize_correlator(data, ref_t=0, time_axis=1)

    np.testing.assert_allclose(out, np.array([[1.0, 2.0, 4.0], [1.0, 2.0, 4.0]]))


def test_preprocess_correlator_pipeline():
    data = np.array(
        [
            [1.0, 2.0, 3.0, 2.0],
            [1.0, np.nan, 3.0, 2.0],
            [2.0, 4.0, 6.0, 4.0],
        ]
    )

    out = preprocess_correlator(
        data,
        sample_axis=0,
        time_axis=1,
        tmin=0,
        tmax=4,
        boundary="periodic",
        ref_t=0,
        drop_invalid=True,
    )

    assert out.shape == (2, 4)
    np.testing.assert_allclose(out[0], np.array([1.0, 1.5, 2.0, 1.5]))


def test_slice_time_extent_rejects_bad_window():
    with pytest.raises(ValueError):
        slice_time_extent(np.arange(6), tmin=4, tmax=2)


def test_average_sources_over_source_axis():
    data = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )

    out = average_sources(data, source_axis=1)

    np.testing.assert_allclose(out, np.array([[2.0, 3.0], [6.0, 7.0]]))


def test_merge_configurations_moves_axis_to_front():
    data = np.arange(12).reshape(3, 4)

    out = merge_configurations(data, config_axis=1)

    np.testing.assert_allclose(out, data.T)


def test_preprocess_nucleon_tmdpdf_source_average_pipeline():
    data = np.array(
        [
            [[1.0, 2.0, 2.0, 1.0], [1.0, 2.0, 2.0, 1.0]],
            [[2.0, 3.0, 3.0, 2.0], [2.0, 3.0, 3.0, 2.0]],
            [[3.0, 4.0, 4.0, 3.0], [3.0, 4.0, 4.0, 3.0]],
        ]
    )

    out = preprocess_nucleon_tmdpdf(
        data,
        config_axis=0,
        source_axis=1,
        time_axis=1,
        tmin=0,
        tmax=4,
        boundary="periodic",
        ref_t=None,
        drop_invalid=True,
    )

    assert out.shape == (3, 4)
    np.testing.assert_allclose(out[1], np.array([2.0, 3.0, 3.0, 2.0]))
