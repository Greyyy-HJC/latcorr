"""Summed-ratio and FH observable helpers from loaded correlator arrays."""

from __future__ import annotations

import numpy as np

from .ratio import get_ratio_data


def get_sum_data(
    pt2: np.ndarray,
    pt3_by_tsep: dict[int, np.ndarray],
    tau_cut: int = 1,
    *,
    sample_axis: int = 0,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Compute summed ratio per tsep after contact-term cuts."""
    ratio_real, ratio_imag = get_ratio_data(
        pt2=pt2,
        pt3_by_tsep=pt3_by_tsep,
        sample_axis=sample_axis,
    )

    sum_real: dict[int, np.ndarray] = {}
    sum_imag: dict[int, np.ndarray] = {}
    for tsep in sorted(ratio_real):
        n_tau = ratio_real[tsep].shape[1]
        start = tau_cut
        stop = n_tau - tau_cut
        if start >= stop:
            raise ValueError(
                f"tau_cut={tau_cut} leaves no tau points for tsep={tsep} with n_tau={n_tau}"
            )
        sum_real[tsep] = np.sum(ratio_real[tsep][:, start:stop], axis=1)
        sum_imag[tsep] = np.sum(ratio_imag[tsep][:, start:stop], axis=1)

    return sum_real, sum_imag


def get_fh_data(
    pt2: np.ndarray,
    pt3_by_tsep: dict[int, np.ndarray],
    tau_cut: int = 1,
    *,
    sample_axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute FH from neighboring tsep summed-ratio differences.

    Returns arrays with shape ``(n_sample, n_tsep-1)``.
    """
    sum_real, sum_imag = get_sum_data(
        pt2=pt2,
        pt3_by_tsep=pt3_by_tsep,
        tau_cut=tau_cut,
        sample_axis=sample_axis,
    )
    tsep_sorted = sorted(sum_real)
    if len(tsep_sorted) < 2:
        raise ValueError("get_fh_data requires at least two tsep entries in pt3_by_tsep")

    fh_real_cols = []
    fh_imag_cols = []
    for t0, t1 in zip(tsep_sorted[:-1], tsep_sorted[1:]):
        dt = t1 - t0
        if dt <= 0:
            raise ValueError("tsep keys must be strictly increasing")
        fh_real_cols.append((sum_real[t1] - sum_real[t0]) / dt)
        fh_imag_cols.append((sum_imag[t1] - sum_imag[t0]) / dt)

    return np.stack(fh_real_cols, axis=1), np.stack(fh_imag_cols, axis=1)
