"""Basic preprocessing utilities for correlator arrays.

The helpers here are intentionally small and composable. They operate on plain
NumPy arrays so they can be reused before resampling, plotting, or fitting.
"""

from __future__ import annotations

import numpy as np


def average_sources(
    data: np.ndarray,
    source_axis: int = 1,
    *,
    config_axis: int = 0,
) -> np.ndarray:
    """Average source measurements within each configuration.

    Parameters
    ----------
    data:
        Array containing a configuration axis and a source axis.
        A common layout is ``(n_cfg, n_src, ...)``.
    source_axis:
        Axis holding the different source measurements for each configuration.
    config_axis:
        Axis holding the configuration index. The returned array keeps this
        axis in front so downstream code can treat it as the sample axis.

    Returns
    -------
    numpy.ndarray
        Array with the source axis averaged out.
    """
    arr = np.asarray(data)
    cfg_axis = config_axis % arr.ndim
    src_axis = source_axis % arr.ndim

    if cfg_axis == src_axis:
        raise ValueError("config_axis and source_axis must be different")

    front_axes = [cfg_axis, src_axis]
    other_axes = [idx for idx in range(arr.ndim) if idx not in front_axes]
    arr = np.transpose(arr, axes=front_axes + other_axes)
    return np.mean(arr, axis=1)


def merge_configurations(data: np.ndarray, config_axis: int = 0) -> np.ndarray:
    """Gather per-configuration data into a single ensemble array.

    In the common case where the input is already a NumPy array with
    configurations on one axis, this function simply moves that axis to the
    front so later resampling code can treat it uniformly.
    """
    arr = np.asarray(data)
    axis = config_axis % arr.ndim
    return np.moveaxis(arr, axis, 0)


def drop_nonfinite_samples(data: np.ndarray, sample_axis: int = 0) -> np.ndarray:
    """Remove samples that contain NaN or inf values."""
    arr = np.asarray(data)
    arr = np.moveaxis(arr, sample_axis, 0)

    if arr.ndim == 1:
        mask = np.isfinite(arr)
    else:
        mask = np.all(np.isfinite(arr), axis=tuple(range(1, arr.ndim)))

    return np.moveaxis(arr[mask], 0, sample_axis)


def slice_time_extent(
    data: np.ndarray,
    tmin: int | None = None,
    tmax: int | None = None,
    *,
    time_axis: int = -1,
) -> np.ndarray:
    """Select a time window from a correlator array."""
    arr = np.asarray(data)
    axis = time_axis % arr.ndim

    start = 0 if tmin is None else tmin
    stop = arr.shape[axis] if tmax is None else tmax
    if start < 0 or stop < 0:
        raise ValueError("tmin and tmax must be non-negative")
    if start >= stop:
        raise ValueError("tmin must be smaller than tmax")

    slc = [slice(None)] * arr.ndim
    slc[axis] = slice(start, stop)
    return arr[tuple(slc)]


def symmetrize_correlator(
    data: np.ndarray,
    *,
    boundary: str = "periodic",
    time_axis: int = -1,
) -> np.ndarray:
    """Symmetrize a correlator using forward/backward time reflection."""
    arr = np.asarray(data)
    axis = time_axis % arr.ndim
    arr_t = np.moveaxis(arr, axis, -1)
    n_t = arr_t.shape[-1]

    if n_t < 2:
        raise ValueError("symmetrize_correlator needs at least two time slices")
    if boundary not in {"periodic", "anti-periodic"}:
        raise ValueError(f"unsupported boundary mode: {boundary!r}")

    reflected = np.flip(arr_t, axis=-1)
    if boundary == "anti-periodic":
        reflected = -reflected

    out = 0.5 * (arr_t + reflected)
    return np.moveaxis(out, -1, axis)


def normalize_correlator(
    data: np.ndarray,
    ref_t: int = 0,
    *,
    time_axis: int = -1,
) -> np.ndarray:
    """Normalize each sample by a reference time slice."""
    arr = np.asarray(data)
    axis = time_axis % arr.ndim
    arr_t = np.moveaxis(arr, axis, -1)

    if not (0 <= ref_t < arr_t.shape[-1]):
        raise ValueError("ref_t is out of range for the time axis")

    ref = arr_t[..., ref_t]
    if np.any(ref == 0):
        raise ZeroDivisionError("reference time slice contains zeros")

    norm = arr_t / ref[..., None]
    return np.moveaxis(norm, -1, axis)


def preprocess_correlator(
    data: np.ndarray,
    *,
    sample_axis: int = 0,
    time_axis: int = -1,
    tmin: int | None = None,
    tmax: int | None = None,
    boundary: str = "periodic",
    ref_t: int | None = None,
    drop_invalid: bool = True,
) -> np.ndarray:
    """Run a small preprocessing pipeline on a correlator array."""
    arr = np.asarray(data)

    if drop_invalid:
        arr = drop_nonfinite_samples(arr, sample_axis=sample_axis)

    if tmin is not None or tmax is not None:
        arr = slice_time_extent(arr, tmin=tmin, tmax=tmax, time_axis=time_axis)

    arr = symmetrize_correlator(arr, boundary=boundary, time_axis=time_axis)

    if ref_t is not None:
        arr = normalize_correlator(arr, ref_t=ref_t, time_axis=time_axis)

    return arr


def preprocess_nucleon_tmdpdf(
    data: np.ndarray,
    *,
    config_axis: int = 0,
    source_axis: int = 1,
    time_axis: int = -1,
    tmin: int | None = None,
    tmax: int | None = None,
    boundary: str = "periodic",
    ref_t: int | None = None,
    drop_invalid: bool = True,
) -> np.ndarray:
    """Preprocess nucleon TMDPDF correlators with source averaging.

    This helper is the natural entry point for raw 2pt/3pt ensemble data that
    still carries a source axis. It first averages over sources within each
    configuration, then applies the generic correlator preprocessing pipeline.
    """
    arr = np.asarray(data)

    if arr.ndim < 3:
        raise ValueError(
            "preprocess_nucleon_tmdpdf expects at least a config axis, source axis, "
            "and one data axis"
        )

    arr = average_sources(arr, source_axis=source_axis, config_axis=config_axis)
    arr = merge_configurations(arr, config_axis=0)

    if drop_invalid:
        arr = drop_nonfinite_samples(arr, sample_axis=0)

    if tmin is not None or tmax is not None:
        arr = slice_time_extent(arr, tmin=tmin, tmax=tmax, time_axis=time_axis)

    arr = symmetrize_correlator(arr, boundary=boundary, time_axis=time_axis)

    if ref_t is not None:
        arr = normalize_correlator(arr, ref_t=ref_t, time_axis=time_axis)

    return arr
