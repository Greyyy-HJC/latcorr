"""Plot helpers for 2pt, ratio, and FH correlator observables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from latcorr.correlators import pt2_to_meff

from .plot_settings import ERRORBAR_STYLE, MEFF_LABEL, TICK_LABEL_STYLE, default_plot


def pt2_plot(
    pt2_resampled: np.ndarray,
    *,
    sample_axis: int = 0,
    boundary: str = "none",
    tmax: int | None = None,
    save_prefix: str | None = None,
    out_dir: str | Path | None = None,
    show: bool = False,
) -> tuple[tuple[Figure, Axes], tuple[Figure, Axes]]:
    """Plot C2pt (log scale) and effective mass from resampled pt2 data."""
    pt2_st = _as_sample_t(pt2_resampled, sample_axis=sample_axis, name="pt2_resampled")
    pt2_ts = np.swapaxes(pt2_st, 0, 1)
    n_t = pt2_ts.shape[0]
    limit = n_t if tmax is None else min(tmax, n_t)
    t = np.arange(limit)

    c2_mean = np.mean(np.real(pt2_ts[:limit]), axis=1)
    c2_err = np.std(np.real(pt2_ts[:limit]), axis=1, ddof=1)

    fig_c2, ax_c2 = default_plot()
    ax_c2.errorbar(t, c2_mean, yerr=c2_err, **ERRORBAR_STYLE)
    ax_c2.set_yscale("log")
    ax_c2.set_xlabel(r"$t~/~a$")
    ax_c2.set_ylabel(r"$C_{2\mathrm{pt}}(t)$")
    ax_c2.tick_params(**TICK_LABEL_STYLE)

    meff_input = np.abs(pt2_ts) if boundary == "none" else np.real(pt2_ts)
    meff_samples = np.array([pt2_to_meff(meff_input[:, i], boundary=boundary) for i in range(pt2_ts.shape[1])])
    meff_mean = np.mean(meff_samples, axis=0)
    meff_err = np.std(meff_samples, axis=0, ddof=1)
    meff_t = np.arange(meff_mean.size)
    if tmax is not None:
        meff_limit = min(tmax, meff_mean.size)
        meff_t = meff_t[:meff_limit]
        meff_mean = meff_mean[:meff_limit]
        meff_err = meff_err[:meff_limit]

    fig_meff, ax_meff = default_plot()
    ax_meff.errorbar(meff_t, meff_mean, yerr=meff_err, **ERRORBAR_STYLE)
    ax_meff.set_xlabel(r"$t~/~a$")
    ax_meff.set_ylabel(MEFF_LABEL)
    ax_meff.tick_params(**TICK_LABEL_STYLE)

    if save_prefix and out_dir is not None:
        output = Path(out_dir)
        output.mkdir(parents=True, exist_ok=True)
        fig_c2.savefig(
            output / f"{save_prefix}_log.pdf",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
        fig_meff.savefig(
            output / f"{save_prefix}_meff.pdf",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )

    if show:
        fig_c2.show()
        fig_meff.show()

    return (fig_c2, ax_c2), (fig_meff, ax_meff)


def ratio_plot(
    ratio_real: dict[int, np.ndarray],
    ratio_imag: dict[int, np.ndarray] | None = None,
    *,
    sample_axis: int = 0,
    tsep_order: list[int] | None = None,
    part: str = "real",
    tmax: int | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot ratio vs tau for each tsep from precomputed arrays."""
    ratio_data = _choose_part(ratio_real, ratio_imag, part=part)
    order = sorted(ratio_data) if tsep_order is None else tsep_order

    fig, ax = default_plot()
    for tsep in order:
        data = _as_sample_tau(ratio_data[tsep], sample_axis=sample_axis, name=f"ratio[{tsep}]")
        n_tau = data.shape[1]
        limit = n_tau if tmax is None else min(tmax, n_tau)
        x = np.arange(limit)
        y = np.mean(data[:, :limit], axis=0)
        yerr = np.std(data[:, :limit], axis=0, ddof=1)
        ax.errorbar(x, y, yerr=yerr, label=f"tsep={tsep}", **ERRORBAR_STYLE)

    ax.set_xlabel(r"$\tau~/~a$")
    ax.set_ylabel(f"ratio ({part})")
    ax.legend()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight", transparent=True)
    if show:
        fig.show()
    return fig, ax


def fh_plot(
    fh_real: np.ndarray,
    fh_imag: np.ndarray | None = None,
    *,
    sample_axis: int = 0,
    tsep: list[int] | np.ndarray | None = None,
    part: str = "real",
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot FH vs tsep from precomputed arrays."""
    data = _choose_part(fh_real, fh_imag, part=part)
    fh_data = _as_sample_tau(data, sample_axis=sample_axis, name="fh")
    y = np.mean(fh_data, axis=0)
    yerr = np.std(fh_data, axis=0, ddof=1)

    if tsep is None:
        x = np.arange(fh_data.shape[1])
    else:
        tsep_arr = np.asarray(tsep)
        if tsep_arr.size == fh_data.shape[1] + 1:
            x = tsep_arr[:-1]
        elif tsep_arr.size == fh_data.shape[1]:
            x = tsep_arr
        else:
            raise ValueError("tsep length must equal n_tsep or n_tsep+1 for fh_plot")

    fig, ax = default_plot()
    ax.errorbar(x, y, yerr=yerr, **ERRORBAR_STYLE)
    ax.set_xlabel(r"$t_{\mathrm{sep}}~/~a$")
    ax.set_ylabel(f"FH ({part})")

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight", transparent=True)
    if show:
        fig.show()
    return fig, ax


def _as_sample_t(data: np.ndarray, *, sample_axis: int, name: str) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}")
    axis = sample_axis % arr.ndim
    return np.moveaxis(arr, axis, 0)


def _as_sample_tau(data: np.ndarray, *, sample_axis: int, name: str) -> np.ndarray:
    return _as_sample_t(data, sample_axis=sample_axis, name=name)


def _choose_part(
    real_data: np.ndarray | dict[int, np.ndarray],
    imag_data: np.ndarray | dict[int, np.ndarray] | None,
    *,
    part: str,
) -> np.ndarray | dict[int, np.ndarray]:
    if part == "real":
        return real_data
    if part == "imag":
        if imag_data is None:
            raise ValueError("imaginary part requested but imag_data is None")
        return imag_data
    raise ValueError("part must be either 'real' or 'imag'")
