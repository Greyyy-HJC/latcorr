"""Strip pion soft form factor source files.

Edit the parameter block below for a different dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_SRC = Path(__file__).resolve().parents[2] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from latcorr.preprocess import strip_pion_softff


#
# Parameter block: edit these values for a different ensemble.
#

ENSEMBLE_TAG = "HISQa076_140MeV"
MOMENTUM_TAG = "ex_wall_PX0PY0PZ9_SP"
INPUT_ROOT = Path("/global/cfs/projectdirs/nplatd/Corr_T0/l64c64a076/qTMD_softFF/RAW/ff")
OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "example_output"
    / "pion_softff"
    / ENSEMBLE_TAG
    / MOMENTUM_TAG
    / f"softFF_{ENSEMBLE_TAG}_{MOMENTUM_TAG}.h5"
)
SUMMARY_PATH = (
    Path(__file__).resolve().parents[2]
    / "example_output"
    / "pion_softff"
    / ENSEMBLE_TAG
    / MOMENTUM_TAG
    / f"summary_{ENSEMBLE_TAG}_{MOMENTUM_TAG}.txt"
)
SOURCE_FILE_GLOB = f"{ENSEMBLE_TAG}.softFF.*.{MOMENTUM_TAG}.x*y*z*t*.fw_*.bw_*.h5"
OVERWRITE = True


def main() -> None:
    out = strip_pion_softff(
        INPUT_ROOT,
        OUTPUT_PATH,
        source_file_glob=SOURCE_FILE_GLOB,
        overwrite=OVERWRITE,
        summary_path=SUMMARY_PATH,
    )
    print(f"saved stripped pion softFF file: {out}")


if __name__ == "__main__":
    main()
