"""HDF5 readers for qTMDWF distribution-amplitude data."""

from __future__ import annotations

from os import PathLike

import h5py
import numpy as np


def read_qda_h5(
    path: str | PathLike[str],
    source_sink: str = "SP",
    gamma: str = "T5",
    momentum: str = "PX0PY0PZ0",
    b_dir: str = "b_X",
    eta: str = "eta0",
    bT: str | None = None,
    bz: str | None = None,
) -> np.ndarray | dict[str, np.ndarray] | dict[str, dict[str, np.ndarray]]:
    """Read qTMDWF quasi-DA datasets from hierarchical HDF5 groups.

    Parameters
    ----------
    path:
        HDF5 file path.
    source_sink:
        Top-level source/sink group, default ``"SP"``.
    gamma:
        Gamma channel group, default ``"T5"``.
    momentum:
        Momentum group, default ``"PX0PY0PZ0"``.
    b_dir:
        Wilson-line direction group, default ``"b_X"``.
    eta:
        Eta group, default ``"eta0"``.
    bT:
        Optional bT group selector (e.g. ``"bT0"``). If omitted, all ``bT*`` are returned.
    bz:
        Optional bz dataset selector (e.g. ``"bz0"``). Requires ``bT``.

    Returns
    -------
    numpy.ndarray or dict
        - If ``bT`` and ``bz`` are both provided: one dataset array.
        - If only ``bT`` is provided: ``dict[bz, array]``.
        - If neither is provided: ``dict[bT, dict[bz, array]]``.
    """
    if bz is not None and bT is None:
        raise ValueError("bT must be provided when bz is specified")

    with h5py.File(path, "r") as h5f:
        base = _require_group(h5f, source_sink, "source_sink")
        base = _require_group(base, gamma, "gamma")
        base = _require_group(base, momentum, "momentum")
        base = _require_group(base, b_dir, "b_dir")
        eta_group = _require_group(base, eta, "eta")

        if bT is None:
            return {bt_key: _read_dataset_group(eta_group[bt_key]) for bt_key in eta_group.keys()}

        bt_group = _require_group(eta_group, bT, "bT")
        if bz is None:
            return _read_dataset_group(bt_group)

        if bz not in bt_group:
            raise KeyError(f"bz key not found under {bT!r}: {bz!r}")

        return np.asarray(bt_group[bz])


def _read_dataset_group(group: h5py.Group) -> dict[str, np.ndarray]:
    """Read all datasets in a group into NumPy arrays."""
    return {key: np.asarray(group[key]) for key in group.keys()}


def _require_group(parent: h5py.File | h5py.Group, key: str, level: str) -> h5py.Group:
    """Return required child group or raise a descriptive error."""
    if key not in parent:
        raise KeyError(f"{level} key not found: {key!r}")

    child = parent[key]
    if not isinstance(child, h5py.Group):
        raise KeyError(f"{level} key is not a group: {key!r}")

    return child
