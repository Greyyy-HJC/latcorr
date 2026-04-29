"""Bootstrap and jackknife resampling interfaces."""

import numpy as np


def bin_data(data: np.ndarray, bin_size: int, axis: int = 0) -> np.ndarray:
    """Average adjacent configurations into bins.

    Parameters
    ----------
    data:
        Input ensemble data.
    bin_size:
        Number of configurations per bin.
    axis:
        Configuration axis.

    Returns
    -------
    numpy.ndarray
        Binned data.
    """
    data = np.asarray(data)

    if bin_size < 1:
        raise ValueError("bin_size must be a positive integer")

    data = np.moveaxis(data, axis, 0)
    n_bins = data.shape[0] // bin_size
    data = data[: n_bins * bin_size]
    data = data.reshape(n_bins, bin_size, *data.shape[1:]).mean(axis=1)

    return np.moveaxis(data, 0, axis)


def bootstrap(
    data: np.ndarray,
    n_samples: int,
    sample_size: int | None = None,
    axis: int = 0,
    bin_size: int = 1,
    seed: int | None = 1984,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Generate bootstrap samples from ensemble data.

    Parameters
    ----------
    data:
        Input ensemble data.
    n_samples:
        Number of bootstrap samples to generate.
    sample_size:
        Number of configurations drawn per bootstrap sample. Defaults to the
        number of configurations.
    axis:
        Configuration axis.
    bin_size:
        Optional bin size applied before resampling.
    seed:
        Random seed for reproducible sampling.
    return_indices:
        Whether to return the sampled configuration indices with the samples.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, numpy.ndarray]
        Bootstrap sample averages, optionally with sampled indices.
    """
    data = np.asarray(data)

    if bin_size > 1:
        data = bin_data(data, bin_size, axis=axis)

    n_conf = data.shape[axis]

    if sample_size is None:
        sample_size = n_conf

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_conf, (n_samples, sample_size), replace=True)
    samples = np.take(data, indices, axis=axis).mean(axis=axis + 1)

    if return_indices:
        return samples, indices

    return samples


def jackknife(data: np.ndarray, axis: int = 0, bin_size: int = 1) -> np.ndarray:
    """Generate leave-one-bin-out jackknife samples.

    Parameters
    ----------
    data:
        Input ensemble data.
    axis:
        Configuration axis.
    bin_size:
        Optional bin size applied before jackknife resampling.

    Returns
    -------
    numpy.ndarray
        Jackknife sample averages.
    """
    data = np.asarray(data)

    if bin_size > 1:
        data = bin_data(data, bin_size, axis=axis)

    n_conf = data.shape[axis]

    if n_conf < 2:
        raise ValueError("jackknife needs at least two samples")

    total = data.sum(axis=axis, keepdims=True)

    return (total - data) / (n_conf - 1)
