"""Example preprocessing workflows for correlator arrays.

This script combines the original basic correlator demo with the synthetic
Nucleon TMDPDF demo so the example folder has a single entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


#
# Basic correlator preprocessing parameters.
#

PROJECT_SRC = Path(__file__).resolve().parents[2] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from latcorr.preprocess import preprocess_correlator, preprocess_nucleon_tmdpdf

BASIC_N_SAMPLE = 8
BASIC_N_T = 12
BASIC_TMIN = 1
BASIC_TMAX = 10
BASIC_BOUNDARY = "periodic"
BASIC_REF_T = 1
BASIC_DROP_INVALID = True


#
# Nucleon TMDPDF preprocessing parameters.
#

TMDPDF_N_CFG = 4
TMDPDF_N_SRC = 3
TMDPDF_N_T = 8
TMDPDF_N_TSEP = 2
TMDPDF_TMIN = 1
TMDPDF_TMAX = 7
TMDPDF_BOUNDARY = "periodic"
TMDPDF_REF_T = 1
TMDPDF_DROP_INVALID = True


def make_basic_correlator(n_sample: int = BASIC_N_SAMPLE, n_t: int = BASIC_N_T) -> np.ndarray:
    """Build a noisy exponential correlator with one invalid sample."""
    rng = np.random.default_rng(1234)
    t = np.arange(n_t)
    base = np.exp(-0.35 * t)
    noise = 0.03 * rng.normal(size=(n_sample, n_t))
    data = base[None, :] * (1.0 + noise)
    data[2, 5] = np.nan
    return data


def make_synthetic_nucleon_tmdpdf_2pt(
    n_cfg: int = TMDPDF_N_CFG,
    n_src: int = TMDPDF_N_SRC,
    n_t: int = TMDPDF_N_T,
) -> np.ndarray:
    """Build a synthetic 2pt-like TMDPDF array with a source axis."""
    rng = np.random.default_rng(2024)
    t = np.arange(n_t)
    base = np.exp(-0.4 * t)
    noise = 0.02 * rng.normal(size=(n_cfg, n_src, n_t))
    return base[None, None, :] * (1.0 + noise)


def make_synthetic_nucleon_tmdpdf_3pt(
    n_cfg: int = TMDPDF_N_CFG,
    n_src: int = TMDPDF_N_SRC,
    n_t: int = TMDPDF_N_T,
    n_tsep: int = TMDPDF_N_TSEP,
) -> np.ndarray:
    """Build a synthetic 3pt-like TMDPDF array with a source axis."""
    rng = np.random.default_rng(2025)
    base = np.exp(-0.2 * np.arange(n_t))[None, None, :, None]
    noise = 0.02 * rng.normal(size=(n_cfg, n_src, n_t, n_tsep))
    data = base * (1.0 + noise)
    data[1, 1, 3, 0] = np.nan
    return data


def demo_basic_preprocess() -> None:
    data = make_basic_correlator()
    cleaned = preprocess_correlator(
        data,
        sample_axis=0,
        time_axis=1,
        tmin=BASIC_TMIN,
        tmax=BASIC_TMAX,
        boundary=BASIC_BOUNDARY,
        ref_t=BASIC_REF_T,
        drop_invalid=BASIC_DROP_INVALID,
    )

    print("basic input shape:", data.shape)
    print("basic output shape:", cleaned.shape)
    print("basic first processed sample:", np.round(cleaned[0], 6))


def demo_nucleon_tmdpdf_preprocess() -> None:
    pt2 = make_synthetic_nucleon_tmdpdf_2pt()
    pt2_processed = preprocess_nucleon_tmdpdf(
        pt2,
        config_axis=0,
        source_axis=1,
        time_axis=1,
        tmin=TMDPDF_TMIN,
        tmax=TMDPDF_TMAX,
        boundary=TMDPDF_BOUNDARY,
        ref_t=TMDPDF_REF_T,
        drop_invalid=TMDPDF_DROP_INVALID,
    )

    pt3 = make_synthetic_nucleon_tmdpdf_3pt()
    pt3_processed = preprocess_nucleon_tmdpdf(
        pt3,
        config_axis=0,
        source_axis=1,
        time_axis=1,
        tmin=TMDPDF_TMIN,
        tmax=TMDPDF_TMAX,
        boundary=TMDPDF_BOUNDARY,
        ref_t=TMDPDF_REF_T,
        drop_invalid=TMDPDF_DROP_INVALID,
    )

    print("tmdpdf 2pt raw shape:", pt2.shape)
    print("tmdpdf 2pt processed shape:", pt2_processed.shape)
    print("tmdpdf 3pt raw shape:", pt3.shape)
    print("tmdpdf 3pt processed shape:", pt3_processed.shape)


def main() -> None:
    demo_basic_preprocess()
    demo_nucleon_tmdpdf_preprocess()


if __name__ == "__main__":
    main()
