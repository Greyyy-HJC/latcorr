import numpy as np
import pytest

from latcorr.resampling import bin_data, bootstrap, jackknife


def test_bin_data_axis0():
    data = np.array([1.0, 3.0, 5.0, 7.0])

    out = bin_data(data, 2)

    np.testing.assert_allclose(out, np.array([2.0, 6.0]))


def test_bin_data_axis1():
    data = np.array([[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]])

    out = bin_data(data, 2, axis=1)

    np.testing.assert_allclose(out, np.array([[2.0, 6.0], [3.0, 7.0]]))


def test_bootstrap_returns_samples_and_indices():
    data = np.array([1.0, 2.0, 3.0])

    samples, indices = bootstrap(data, 4, seed=123, return_indices=True)

    assert samples.shape == (4,)
    assert indices.shape == (4, 3)
    np.testing.assert_allclose(samples, data[indices].mean(axis=1))


def test_bootstrap_axis1():
    data = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])

    samples, indices = bootstrap(data, 5, axis=1, seed=123, return_indices=True)

    assert samples.shape == (2, 5)
    np.testing.assert_allclose(samples[:, 0], data[:, indices[0]].mean(axis=1))


def test_jackknife_axis0():
    data = np.array([1.0, 2.0, 4.0])

    out = jackknife(data)

    np.testing.assert_allclose(out, np.array([3.0, 2.5, 1.5]))


def test_jackknife_axis1():
    data = np.array([[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]])

    out = jackknife(data, axis=1)

    np.testing.assert_allclose(out, np.array([[3.0, 2.5, 1.5], [6.0, 5.0, 3.0]]))


def test_jackknife_needs_two_samples():
    with pytest.raises(ValueError):
        jackknife(np.array([1.0]))
