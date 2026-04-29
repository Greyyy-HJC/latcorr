"""Shared resampling utilities for correlator readers."""

from __future__ import annotations

from typing import Literal

import numpy as np

from latcorr.resampling import bootstrap, jackknife

ResamplingMode = Literal["none", "jk", "bs"]


def apply_resampling(
    data: np.ndarray,
    mode: ResamplingMode = "none",
    *,
    sample_axis: int = 0,
    n_samples: int = 200,
    bin_size: int = 5,
    seed: int | None = 1984,
) -> np.ndarray:
    """Apply optional resampling on a correlator array."""
    data = np.asarray(data)
    if mode == "none":
        return data
    if mode == "jk":
        return jackknife(data, axis=sample_axis, bin_size=bin_size)
    if mode == "bs":
        return bootstrap(
            data,
            n_samples=n_samples,
            axis=sample_axis,
            bin_size=bin_size,
            seed=seed,
        )
    raise ValueError(f"unsupported resampling mode: {mode!r}")
