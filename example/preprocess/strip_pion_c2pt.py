"""Example: strip pion 2pt source averages from qTMDWF per-cfg folders.

Edit the parameter block below for a different dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_SRC = Path(__file__).resolve().parents[2] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from latcorr.preprocess import strip_pion_c2pt


#
# Parameter block: edit these values for a different ensemble.
#

ENSEMBLE_TAG = "l64c64a076"
FILE_TAG = "HISQa076"
SOURCE_TYPES = ["srcT5"]
MOMENTUM_TAG = "1HYP_M140_GSRC_W45_k0"
CFG_LIST = ["1056", "1062"]

INPUT_ROOT = Path(f"/global/cfs/projectdirs/nplatd/Corr_T0/{ENSEMBLE_TAG}/qTMDWF/CG/c2pt")
OUTPUT_ROOT = Path(
    Path(__file__).resolve().parents[2] / "example_output" / "pion_c2pt" / ENSEMBLE_TAG / MOMENTUM_TAG
)
SOURCE_SINK = "SS"
OVERWRITE = True


def main() -> None:
    cfg_dirs = [INPUT_ROOT / cfg for cfg in CFG_LIST]

    for source_type in SOURCE_TYPES:
        combo_root = OUTPUT_ROOT / source_type
        combo_output = combo_root / f"c2pt_merged_{source_type}_{MOMENTUM_TAG}.h5"
        combo_summary = combo_root / f"summary_{source_type}_{MOMENTUM_TAG}.txt"
        source_file_glob = f"{FILE_TAG}.c2pt.*.{MOMENTUM_TAG}.{source_type}.h5"

        out = strip_pion_c2pt(
            INPUT_ROOT,
            combo_output,
            cfg_dirs=cfg_dirs,
            source_type=source_type,
            momentum_tag=MOMENTUM_TAG,
            source_file_glob=source_file_glob,
            source_sink=SOURCE_SINK,
            overwrite=OVERWRITE,
            summary_path=combo_summary,
        )
        print(
            f"saved stripped pion c2pt file for {source_type} {MOMENTUM_TAG}: {out}"
        )


if __name__ == "__main__":
    main()
