"""Ratio helpers from already-loaded 2pt/3pt correlator arrays."""

from __future__ import annotations

import numpy as np

def get_ratio_data(
    pt2: np.ndarray,
    pt3_by_tsep: dict[int, np.ndarray],
    *,
    sample_axis: int = 0,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Compute sample-wise ratio for each tsep in ``pt3_by_tsep``.

    Parameters
    ----------
    pt2:
        2-point correlator array. By default, samples are expected on axis 0.
    pt3_by_tsep:
        Dict keyed by integer ``tsep``. Each value must be a 2D array with
        samples on axis 0 by default.
    sample_axis:
        Sample axis for both ``pt2`` and each ``pt3_by_tsep`` entry.
    """
    if not isinstance(pt3_by_tsep, dict) or not pt3_by_tsep:
        raise ValueError("pt3_by_tsep must be a non-empty dict keyed by tsep")
    pt2_sample_t = _as_sample_t(np.asarray(pt2), sample_axis=sample_axis, name="pt2")

    ratio_real: dict[int, np.ndarray] = {}
    ratio_imag: dict[int, np.ndarray] = {}

    for tsep, pt3 in pt3_by_tsep.items():
        if not isinstance(tsep, int):
            raise ValueError("pt3_by_tsep keys must be integer tsep values")
        if pt3.ndim != 2:
            raise ValueError(f"pt3_by_tsep[{tsep}] must be a 2D array")

        pt2_t = _pt2_slice_by_tsep(pt2_sample_t, tsep)
        pt3_conf_tau = _as_conf_tau(
            np.asarray(pt3),
            pt2_t.shape[0],
            sample_axis=sample_axis,
            name=f"pt3_by_tsep[{tsep}]",
        )

        ratio = pt3_conf_tau / pt2_t[:, None]
        ratio_real[tsep] = np.real(ratio)
        ratio_imag[tsep] = np.imag(ratio)

    return ratio_real, ratio_imag


def _pt2_slice_by_tsep(pt2: np.ndarray, tsep: int) -> np.ndarray:
    """Extract ``pt2[tsep]`` as a 1D sample vector."""
    if 0 <= tsep < pt2.shape[1]:
        return pt2[:, tsep]
    raise ValueError(f"tsep index out of range for pt2 array shape {pt2.shape}")


def _as_sample_t(data: np.ndarray, *, sample_axis: int, name: str) -> np.ndarray:
    """Normalize a 2D array to axes (sample, t)."""
    if data.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {data.shape}")
    axis = sample_axis % data.ndim
    return np.moveaxis(data, axis, 0)


def _as_conf_tau(data: np.ndarray, n_sample: int, *, sample_axis: int, name: str) -> np.ndarray:
    """Normalize a 2D array to axes (sample, tau)."""
    arr = _as_sample_t(data, sample_axis=sample_axis, name=name)
    if arr.shape[0] == n_sample:
        return arr
    if arr.shape[1] == n_sample:
        return np.swapaxes(arr, 0, 1)
    raise ValueError(
        f"{name} sample axis mismatch with pt2: got shape {data.shape}, "
        f"expected one axis equal to {n_sample}"
    )
