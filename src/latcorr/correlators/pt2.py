"""HDF5 readers for 2-point correlator data."""
# %%

from __future__ import annotations

from os import PathLike

import h5py
import numpy as np
from scipy.optimize import fsolve

from ._resampling import ResamplingMode, apply_resampling


def read_pt2_h5(
    path: str | PathLike[str],
    source_sink: str = "SS",
    gamma: str | None = None,
    momentum: str | None = None,
    resampling: ResamplingMode = "none",
    n_samples: int = 200,
    bin_size: int = 5,
    seed: int | None = 1984,
) -> np.ndarray | dict[str, np.ndarray] | dict[str, dict[str, np.ndarray]]:
    """Read 2-point correlator datasets from a comb_c2pt HDF5 file.

    Parameters
    ----------
    path:
        HDF5 file path.
    source_sink:
        Top-level group, e.g. ``"SS"``.
    gamma:
        Gamma channel key under ``source_sink`` (e.g. ``"5"``, ``"I"``, ``"T"``).
        If omitted, all available gamma channels are returned.
    momentum:
        Momentum dataset key under ``source_sink/gamma`` (e.g. ``"PX0PY0PZ0"``).
        If provided, ``gamma`` must also be provided.

    resampling:
        Resampling mode: ``"none"``, ``"jk"``, or ``"bs"``.
    n_samples:
        Number of bootstrap samples when ``resampling="bs"``.
    bin_size:
        Optional binning size before jackknife/bootstrap.
    seed:
        Random seed for bootstrap sampling.

    Returns
    -------
    numpy.ndarray or dict
        - If ``gamma`` and ``momentum`` are both provided: one dataset array.
        - If only ``gamma`` is provided: ``dict[momentum, array]``.
        - If neither is provided: ``dict[gamma, dict[momentum, array]]``.
    """
    if momentum is not None and gamma is None:
        raise ValueError("gamma must be provided when momentum is specified")

    with h5py.File(path, "r") as h5f:
        if source_sink not in h5f:
            raise KeyError(f"source_sink group not found: {source_sink!r}")

        source_group = h5f[source_sink]

        if gamma is None:
            return {
                gamma_key: _read_momentum_group(
                    source_group[gamma_key],
                    resampling=resampling,
                    n_samples=n_samples,
                    bin_size=bin_size,
                    seed=seed,
                )
                for gamma_key in source_group.keys()
            }

        if gamma not in source_group:
            raise KeyError(f"gamma key not found under {source_sink!r}: {gamma!r}")

        gamma_group = source_group[gamma]

        if momentum is None:
            return _read_momentum_group(
                gamma_group,
                resampling=resampling,
                n_samples=n_samples,
                bin_size=bin_size,
                seed=seed,
            )

        if momentum not in gamma_group:
            raise KeyError(
                f"momentum key not found under {source_sink!r}/{gamma!r}: {momentum!r}"
            )

        data = np.swapaxes(np.asarray(gamma_group[momentum]), 0, 1)
        return apply_resampling(
            data,
            resampling,
            sample_axis=0,
            n_samples=n_samples,
            bin_size=bin_size,
            seed=seed,
        )


def _read_momentum_group(
    group: h5py.Group,
    *,
    resampling: ResamplingMode,
    n_samples: int,
    bin_size: int,
    seed: int | None,
) -> dict[str, np.ndarray]:
    """Read all momentum datasets in a gamma group."""
    return {
        key: apply_resampling(
            np.swapaxes(np.asarray(group[key]), 0, 1),
            resampling,
            sample_axis=0,
            n_samples=n_samples,
            bin_size=bin_size,
            seed=seed,
        )
        for key in group.keys()
    }


def pt2_to_meff(pt2_array: np.ndarray, boundary: str = "periodic") -> np.ndarray:
    """Convert a 1D 2pt correlator to effective-mass values."""
    data = np.asarray(pt2_array)
    if data.ndim != 1:
        raise ValueError(f"pt2_to_meff expects a 1D array, got shape {data.shape}")
    if data.size < 3 and boundary in {"periodic", "anti-periodic"}:
        raise ValueError("pt2_array must have at least 3 points for periodic/anti-periodic")
    if data.size < 2 and boundary == "none":
        raise ValueError("pt2_array must have at least 2 points for boundary='none'")

    if boundary == "periodic":
        return np.arccosh((data[2:] + data[:-2]) / (2 * data[1:-1]))
    if boundary == "anti-periodic":
        return np.arcsinh((data[2:] + data[:-2]) / (2 * data[1:-1]))
    if boundary == "none":
        return np.log(data[:-1] / data[1:])
    raise ValueError(f"unsupported boundary mode: {boundary!r}")


def pt2_to_meff_solve(pt2_array: np.ndarray, boundary: str = "periodic") -> np.ndarray:
    """Convert a 1D 2pt correlator to effective mass by solving the ratio equation."""
    data = np.asarray(pt2_array)
    if data.ndim != 1:
        raise ValueError(f"pt2_to_meff_solve expects a 1D array, got shape {data.shape}")
    if data.size < 2 and boundary == "none":
        raise ValueError("pt2_array must have at least 2 points for boundary='none'")
    if data.size < 2 and boundary in {"periodic", "anti-periodic"}:
        raise ValueError("pt2_array must have at least 2 points")

    if boundary == "none":
        return np.log(data[:-1] / data[1:])
    if boundary not in {"periodic", "anti-periodic"}:
        raise ValueError(f"unsupported boundary mode: {boundary!r}")

    nt = data.size
    meff_values: list[float] = []

    def equation(meff: float, t: int, ct: complex, ctp1: complex) -> complex:
        if boundary == "periodic":
            return ct * np.cosh(meff * (t + 1 - nt / 2)) - ctp1 * np.cosh(meff * (t - nt / 2))
        return ct * np.sinh(meff * (t + 1 - nt / 2)) - ctp1 * np.sinh(meff * (t - nt / 2))

    for t in range(nt - 1):
        ct = data[t]
        ctp1 = data[t + 1]
        solution = fsolve(equation, 1.0, args=(t, ct, ctp1))
        meff_values.append(float(np.real(solution[0])))

    return np.asarray(meff_values)
