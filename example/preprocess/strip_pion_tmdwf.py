"""Example: strip pion TMDWF source averages from per-configuration folders.

Edit the parameter block below for a different dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_SRC = Path(__file__).resolve().parents[2] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from latcorr.preprocess import average_pion_tmdwf_cfgs, combine_pion_tmdwf_cfg_files


#
# Parameter block: edit these values for a different ensemble.
#

ENSEMBLE_TAG = "l64c64a076"
MOMENTUM_TAG = "1HYP_M140_GSRC_W45_k0"
SOURCE_TYPE = "src5"
OPERATOR_TAG = "OT5"
CFG_LIST = ["1050", "1056"]

INPUT_ROOT = Path(f"/global/cfs/projectdirs/nplatd/Corr_T0/{ENSEMBLE_TAG}/qTMDWF/CG/qTMDWF")
OUTPUT_ROOT = Path(
    Path(__file__).resolve().parents[2] / "example_output" / "pion_tmdwf" / ENSEMBLE_TAG / MOMENTUM_TAG
)
CFG_OUTPUT_ROOT = OUTPUT_ROOT / "cfg"
FINAL_OUTPUT_PATH = OUTPUT_ROOT / f"qTMDWF_CG_{MOMENTUM_TAG}_{SOURCE_TYPE}_{OPERATOR_TAG}.h5"
SUMMARY_PATH = OUTPUT_ROOT / f"summary_{SOURCE_TYPE}_{OPERATOR_TAG}.txt"

SOURCE_SINK = "SP"
FILE_TAG = "HISQa076"
SOURCE_FILE_GLOB = f"{FILE_TAG}.qTMDWF.*.{MOMENTUM_TAG}.{SOURCE_TYPE}.{OPERATOR_TAG}.h5"
OVERWRITE = True


def main() -> None:
    cfg_dirs = [INPUT_ROOT / cfg for cfg in CFG_LIST]

    average_pion_tmdwf_cfgs(
        INPUT_ROOT,
        CFG_OUTPUT_ROOT,
        cfg_dirs=cfg_dirs,
        source_sink=SOURCE_SINK,
        source_file_glob=SOURCE_FILE_GLOB,
        overwrite=OVERWRITE,
        summary_path=SUMMARY_PATH,
    )
    out = combine_pion_tmdwf_cfg_files(
        CFG_OUTPUT_ROOT,
        FINAL_OUTPUT_PATH,
        source_sink=SOURCE_SINK,
        overwrite=OVERWRITE,
        summary_path=SUMMARY_PATH,
    )
    print(f"saved stripped pion tmdwf file: {out}")


if __name__ == "__main__":
    main()
