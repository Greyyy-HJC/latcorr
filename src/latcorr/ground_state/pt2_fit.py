"""Two-point correlator ground-state fits."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import gvar as gv
import lsqfit as lsf
import numpy as np

from .fit_funcs import general_prior, pt2_re_fcn

logger = logging.getLogger(__name__)


def _trange_to_array(
    trange: tuple[int, int] | Sequence[int] | np.ndarray,
    *,
    data_length: int | None = None,
) -> np.ndarray:
    if isinstance(trange, tuple):
        if len(trange) != 2:
            raise ValueError("trange tuple must be (tmin, tmax)")
        tmin, tmax = trange
        values = np.arange(tmin, tmax, dtype=int)
    else:
        values = np.asarray(trange, dtype=int)

    if values.ndim != 1:
        raise ValueError(f"trange must be one-dimensional, got shape {values.shape}")
    if values.size == 0:
        raise ValueError("trange must contain at least one time slice")
    if np.any(values < 0):
        raise ValueError("trange must not contain negative time slices")
    if data_length is not None and np.any(values >= data_length):
        raise ValueError(
            f"trange contains time slices outside data length {data_length}: {values}"
        )

    return values


def _pt2_prior(nstate: int) -> gv.BufferDict:
    prior = general_prior(nstate=nstate)
    pt2_prior = gv.BufferDict()
    pt2_prior["E0"] = prior["E0"]
    for state in range(1, nstate):
        pt2_prior[f"log(dE{state})"] = prior[f"log(dE{state})"]
    for state in range(nstate):
        pt2_prior[f"z{state}"] = prior[f"z{state}"]
    return pt2_prior


def pt2_fit(
    pt2_avg: np.ndarray,
    trange: tuple[int, int] | Sequence[int] | np.ndarray,
    Lt: int,
    *,
    nstate: int = 2,
    prior: gv.BufferDict | dict[str, gv.GVar] | None = None,
    normalize: bool = True,
    label: str | None = None,
    maxit: int = 10000,
) -> lsf.nonlinear_fit:
    """Fit a two-point correlator with an n-state spectral decomposition."""

    pt2_avg = np.asarray(pt2_avg)
    if pt2_avg.ndim != 1:
        raise ValueError(f"pt2_avg must be one-dimensional, got shape {pt2_avg.shape}")

    fit_t = _trange_to_array(trange, data_length=len(pt2_avg))
    priors = _pt2_prior(nstate=nstate) if prior is None else prior

    normalization_factor = 1.0
    if normalize:
        normalization_factor = abs(pt2_avg[0].mean)
        if normalization_factor == 0:
            raise ZeroDivisionError("cannot normalize pt2 data with zero t=0 mean")
        fit_pt2 = pt2_avg[fit_t] / normalization_factor
    else:
        fit_pt2 = pt2_avg[fit_t]

    def fcn(t: np.ndarray, p: dict) -> np.ndarray:
        return pt2_re_fcn(t, p, Lt, nstate=nstate)

    fit_res = lsf.nonlinear_fit(
        data=(fit_t, fit_pt2),
        prior=priors,
        fcn=fcn,
        maxit=maxit,
    )

    fit_res.trange = fit_t
    fit_res.Lt = Lt
    fit_res.nstate = nstate
    fit_res.normalize = normalize
    fit_res.normalization_factor = normalization_factor
    fit_res.label = label

    fit_label = f" {label}" if label else ""
    fit_quality = f"Q = {fit_res.Q:.3f}, Chi2/dof = {fit_res.chi2 / fit_res.dof:.3f}"
    if fit_res.Q < 0.05:
        logger.warning("Bad 2pt%s fit with %s", fit_label, fit_quality)
    else:
        logger.info("Good 2pt%s fit with %s", fit_label, fit_quality)

    return fit_res


def pt2_two_state_fit(
    pt2_avg: np.ndarray,
    trange: tuple[int, int] | Sequence[int] | np.ndarray,
    Lt: int,
    *,
    prior: gv.BufferDict | dict[str, gv.GVar] | None = None,
    normalize: bool = True,
    label: str | None = None,
    maxit: int = 10000,
) -> lsf.nonlinear_fit:
    """Fit a two-point correlator with the default two-state model."""

    return pt2_fit(
        pt2_avg,
        trange,
        Lt,
        nstate=2,
        prior=prior,
        normalize=normalize,
        label=label,
        maxit=maxit,
    )
