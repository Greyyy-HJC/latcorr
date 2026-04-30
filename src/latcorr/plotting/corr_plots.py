"""Plot helpers for 2pt, ratio, and FH correlator observables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import gvar as gv
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from latcorr.correlators import pt2_to_meff

from .plot_settings import ERRORBAR_STYLE, ERRORBAR_CIRCLE_STYLE, FONT_SIZE, LEGEND_SIZE, MEFF_LABEL, RATIO_REAL_LABEL, RATIO_IMAG_LABEL, TSEP, TSEP_LABEL, TAU_CENTER_LABEL, auto_ylim, default_plot


def pt2_plot(
    pt2_gv_ls: list[np.ndarray],
    boundary: str = "none",
    tmin: int | None = None,
    tmax: int | None = None,
    *,
    save_prefix: str | None = None,
    out_dir: str | Path | None = None,
    show: bool = False,
) -> tuple[tuple[Figure, Axes], tuple[Figure, Axes]]:
    """Plot C2pt (log scale) and effective mass from resampled pt2 data."""
    
    # check if all pt2_gv arrays have the same length
    lengths = [len(pt2_gv) for pt2_gv in pt2_gv_ls]
    if not all(l == lengths[0] for l in lengths):
        raise ValueError("All pt2_gv arrays in pt2_gv_ls must have the same length, got lengths: {}".format(lengths))
    
    if tmin is None: tmin = 0
    if tmax is None: tmax = lengths[0]
    
    t = np.arange(tmin, tmax)
    
    fig_c2, ax_c2 = default_plot()
    for pt2_gv in pt2_gv_ls:
        ax_c2.errorbar(t, gv.mean(pt2_gv[t]), yerr=gv.sdev(pt2_gv[t]), **ERRORBAR_STYLE)
    ax_c2.set_yscale("log")
    ax_c2.set_xlabel(TSEP_LABEL, **FONT_SIZE)
    ax_c2.set_ylabel(r"$C_{2\mathrm{pt}}(t_{\mathrm{sep}})$", **FONT_SIZE)

    fig_meff, ax_meff = default_plot()
    for pt2_gv in pt2_gv_ls:
        meff_gv = pt2_to_meff(pt2_gv[t], boundary=boundary)
        ax_meff.errorbar(np.arange(len(meff_gv)), gv.mean(meff_gv), yerr=gv.sdev(meff_gv), **ERRORBAR_STYLE)
    ax_meff.set_xlabel(TSEP_LABEL, **FONT_SIZE)
    ax_meff.set_ylabel(MEFF_LABEL, **FONT_SIZE)

    if save_prefix and out_dir is not None:
        output = Path(out_dir)
        output.mkdir(parents=True, exist_ok=True)
        fig_c2.savefig(
            output / f"{save_prefix}_c2pt.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        fig_meff.savefig(
            output / f"{save_prefix}_meff.pdf",
            bbox_inches="tight",
            transparent=True,
        )

    if show:
        fig_c2.show()
        fig_meff.show()

    return (fig_c2, ax_c2), (fig_meff, ax_meff)


def ratio_plot(
    tau_dict: dict[int, np.ndarray],
    ratio_real: dict[int, np.ndarray],
    ratio_imag: dict[int, np.ndarray] | None = None,
    *,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot ratio vs tau for each tsep from precomputed arrays."""
    
    tsep_ls = sorted(ratio_real.keys())
    
    fig_real, ax_real = default_plot()
    for tsep in tsep_ls:
        ax_real.errorbar(tau_dict[tsep] - tsep/2, gv.mean(ratio_real[tsep]), yerr=gv.sdev(ratio_real[tsep]), label=f"{TSEP}={tsep} $a$", **ERRORBAR_CIRCLE_STYLE)
    ax_real.set_xlabel(TAU_CENTER_LABEL, **FONT_SIZE)
    ax_real.set_ylabel(RATIO_REAL_LABEL, **FONT_SIZE)
    ax_real.legend(ncol=2, loc="upper right", **LEGEND_SIZE)
    ax_real.set_ylim(auto_ylim([gv.mean(ratio_real[tsep]) for tsep in tsep_ls], [gv.sdev(ratio_real[tsep]) for tsep in tsep_ls], 2))
    
    if ratio_imag is not None:
        fig_imag, ax_imag = default_plot()
        for tsep in tsep_ls:
            ax_imag.errorbar(tau_dict[tsep] - tsep/2, gv.mean(ratio_imag[tsep]), yerr=gv.sdev(ratio_imag[tsep]), label=f"{TSEP}={tsep} $a$", **ERRORBAR_CIRCLE_STYLE)
        ax_imag.set_xlabel(TAU_CENTER_LABEL, **FONT_SIZE)
        ax_imag.set_ylabel(RATIO_IMAG_LABEL, **FONT_SIZE)
        ax_imag.legend(ncol=2, loc="upper right", **LEGEND_SIZE)
        ax_imag.set_ylim(auto_ylim([gv.mean(ratio_imag[tsep]) for tsep in tsep_ls], [gv.sdev(ratio_imag[tsep]) for tsep in tsep_ls], 2))
        
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig_real.savefig(path / f"{save_path}_real.pdf", bbox_inches="tight", transparent=True)
            fig_imag.savefig(path / f"{save_path}_imag.pdf", bbox_inches="tight", transparent=True)

        if show:
            fig_real.show()
            fig_imag.show()
    
        return (fig_real, ax_real), (fig_imag, ax_imag)
    
    else:
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig_real.savefig(path / f"{save_path}_real.pdf", bbox_inches="tight", transparent=True)
        if show:
            fig_real.show()
        return (fig_real, ax_real)


def fh_plot(
    tsep_ls: list[int],
    fh_real: np.ndarray,
    fh_imag: np.ndarray | None = None,
    *,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[tuple[Figure, Axes], tuple[Figure, Axes]] | tuple[Figure, Axes]:
    """Plot FH vs tsep from precomputed gvar arrays."""

    fig_real, ax_real = default_plot()
    ax_real.errorbar(tsep_ls, gv.mean(fh_real), yerr=gv.sdev(fh_real), **ERRORBAR_CIRCLE_STYLE)
    ax_real.set_xlabel(TSEP_LABEL, **FONT_SIZE)
    ax_real.set_ylabel(r"$\Re[\mathrm{FH}(t_{\mathrm{sep}})]$", **FONT_SIZE)
    ax_real.set_ylim(auto_ylim([gv.mean(fh_real)], [gv.sdev(fh_real)], 2))

    if fh_imag is not None:
        fig_imag, ax_imag = default_plot()
        ax_imag.errorbar(tsep_ls, gv.mean(fh_imag), yerr=gv.sdev(fh_imag), **ERRORBAR_CIRCLE_STYLE)
        ax_imag.set_xlabel(TSEP_LABEL, **FONT_SIZE)
        ax_imag.set_ylabel(r"$\Im[\mathrm{FH}(t_{\mathrm{sep}})]$", **FONT_SIZE)
        ax_imag.set_ylim(auto_ylim([gv.mean(fh_imag)], [gv.sdev(fh_imag)], 2))

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig_real.savefig(path.with_name(f"{path.name}_real.pdf"), bbox_inches="tight", transparent=True)
            fig_imag.savefig(path.with_name(f"{path.name}_imag.pdf"), bbox_inches="tight", transparent=True)

        if show:
            fig_real.show()
            fig_imag.show()
        return (fig_real, ax_real), (fig_imag, ax_imag)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig_real.savefig(path.with_name(f"{path.name}_real.pdf"), bbox_inches="tight", transparent=True)
    if show:
        fig_real.show()
    return fig_real, ax_real


