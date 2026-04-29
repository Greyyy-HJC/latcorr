"""HDF5 readers for 2-point correlator data."""
# %%

from __future__ import annotations

from os import PathLike

import h5py
import numpy as np


def read_pt2_h5(
    path: str | PathLike[str],
    source_sink: str = "SS",
    gamma: str | None = None,
    momentum: str | None = None,
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
                gamma_key: _read_momentum_group(source_group[gamma_key])
                for gamma_key in source_group.keys()
            }

        if gamma not in source_group:
            raise KeyError(f"gamma key not found under {source_sink!r}: {gamma!r}")

        gamma_group = source_group[gamma]

        if momentum is None:
            return _read_momentum_group(gamma_group)

        if momentum not in gamma_group:
            raise KeyError(
                f"momentum key not found under {source_sink!r}/{gamma!r}: {momentum!r}"
            )

        return np.asarray(gamma_group[momentum])


def _read_momentum_group(group: h5py.Group) -> dict[str, np.ndarray]:
    """Read all momentum datasets in a gamma group."""
    return {key: np.asarray(group[key]) for key in group.keys()}
