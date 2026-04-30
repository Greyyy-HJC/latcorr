"""Example: strip nucleon 2pt source averages from per-configuration folders.

Edit the parameter block below for a different dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_SRC = Path(__file__).resolve().parents[2] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from latcorr.preprocess import strip_nucleon_c2pt


#
# Parameter block: edit these values for a different ensemble.
#

# dataset tags
ENSEMBLE_TAG = "l64c64a076"
MOMENTUM_TAG = "1HYP_GSRC_W90_k0_T5_PX0PY0PZ0"
srctype = "5"
CFG_LIST = ["1050", "1068"]

# file selection
SOURCE_FILE_GLOB = "*.h5"
pxlist = list(range(-2, 2 + 1))
pylist = list(range(-2, 2 + 1))
pzlist = list(range(-2, 2 + 1))
gmlist = [
    "5",
    "T",
    "T5",
    "X",
    "X5",
    "Y",
    "Y5",
    "Z",
    "Z5",
    "I",
    "SXT",
    "SXY",
    "SXZ",
    "SYT",
    "SYZ",
    "SZT",
]

# input and output paths
INPUT_ROOT = Path(
    f"/global/cfs/projectdirs/nplatd/Corr_T0/{ENSEMBLE_TAG}/nucleon_TMD_pyquda/"
    f"{MOMENTUM_TAG}/c2pt_stream_a_{srctype}_k0_src1-32"
)
OUTPUT_ROOT = Path(
    Path(__file__).resolve().parents[2] / "example_output" / "c2pt" / ENSEMBLE_TAG / MOMENTUM_TAG
)
SUMMARY_PATH = OUTPUT_ROOT / f"c2pt_src{srctype}_{MOMENTUM_TAG}.summary.txt"

# derived lists used by the helper
GAMMAS = gmlist
MOMENTA = [f"PX{px}PY{py}PZ{pz}" for px in pxlist for py in pylist for pz in pzlist]
SOURCE_SINK = "SS"
NT = 64
APPLY_SOURCE_TIME_SIGN = True


def main() -> None:
    cfg_dirs = [INPUT_ROOT / cfg for cfg in CFG_LIST]
    out = strip_nucleon_c2pt(
        INPUT_ROOT,
        OUTPUT_ROOT / f"c2pt_src{srctype}_{MOMENTUM_TAG}.h5",
        cfg_dirs=cfg_dirs,
        gammas=GAMMAS,
        momenta=MOMENTA,
        source_sink=SOURCE_SINK,
        source_file_glob=SOURCE_FILE_GLOB,
        nt=NT,
        apply_source_time_sign=APPLY_SOURCE_TIME_SIGN,
        overwrite=True,
        summary_path=SUMMARY_PATH,
    )
    print(f"saved stripped c2pt file: {out}")


if __name__ == "__main__":
    main()
