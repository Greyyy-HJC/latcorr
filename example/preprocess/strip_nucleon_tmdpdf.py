"""Example: two-step stripping for nucleon TMDPDF 3-point source files.

Edit the parameter block below for a different dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_SRC = Path(__file__).resolve().parents[2] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from latcorr.preprocess import average_nucleon_tmdpdf_cfgs, combine_tmdpdf_cfg_files


#
# Parameter block: edit these values for a different ensemble.
# The string fields below feed the legacy-style file and directory tags.
#

# file and directory tags
ENSEMBLE_TAG = "l64c64a076"
MOMENTUM_TAG = "1HYP_GSRC_W90_k0_T5_PX0PY0PZ0"
Nt = 64
tslist = [4] # [4, 6, 8, 10, 12]
projlist = ["PpUnpol"]
projSave = ["PpUnpol"]
TMDtype = ["CG"]
CFG_LIST = ["1050", "1068"]
QUARK_FLAVORS = ["D"] # ["D", "U"]
SOURCE_TAG = "T5"
sm = "SS"
gmlist = ["T"]
# gmlist = ["5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5"]

# momentum-transfer and separation path filters
qxlist = [0]  # [-2, 2+1]
qylist = [0]  # [-2, 2+1]
qzlist = [0]  # [-2, 0+1]
qxyzlist = [[qx, qy, qz] for qx in qxlist for qy in qylist for qz in qzlist]
bTdir = ["b_X", "b_Y"]
etalist = ["0"]
bTlist = {
    "0": [0], # [0, 1, ..., 20]
}
bzlist = {
    "0": [0], # [-20, ..., 20]
}

# input and output paths
OUTPUT_ROOT = Path(
    Path(__file__).resolve().parents[2] / "example_output" / "tmdpdf" / ENSEMBLE_TAG / MOMENTUM_TAG
)

INPUT_ROOT = Path(
    f"/global/cfs/projectdirs/nplatd/Corr_T0/{ENSEMBLE_TAG}/nucleon_TMD_pyquda/"
    f"{MOMENTUM_TAG}/qTMD"
)

SOURCE_SINK = sm

APPLY_SOURCE_TIME_SIGN = True
OVERWRITE = True


def build_selected_path_globs(gm: str) -> list[str]:
    """Build the HDF5 dataset paths to include for one gm selection."""
    globs: list[str] = []
    for qxyz in qxyzlist:
        momentum_path = f"{gm}/PX{qxyz[0]}PY{qxyz[1]}PZ{qxyz[2]}"
        for idir in bTdir:
            for eta in etalist:
                for bT in bTlist[eta]:
                    for bz in bzlist[eta]:
                        globs.append(f"{momentum_path}/{idir}/eta{eta}/bT{bT}/bz{bz}")
    return globs

def main() -> None:
    # The preprocessing helper flattens the qTMD tree generically, but we keep
    # the legacy-style labels here so changing datasets remains obvious.
    _ = (
        Nt,
        tslist,
        projlist,
        projSave,
        TMDtype,
        QUARK_FLAVORS,
        SOURCE_TAG,
        CFG_LIST,
        OUTPUT_ROOT,
        qxlist,
        qylist,
        qzlist,
        qxyzlist,
        bTdir,
        etalist,
        bTlist,
        bzlist,
    )

    for tmd_type in TMDtype:
        for proj, proj_save in zip(projlist, projSave):
            for tsep in tslist:
                for quark_flavor in QUARK_FLAVORS:
                    for gm in gmlist:
                        combo_root = OUTPUT_ROOT / quark_flavor / tmd_type / proj_save / f"dt{tsep}" / gm
                        combo_cfg_root = combo_root / "cfg"
                        combo_final_output = (
                            combo_root
                            / f"qTMD_src{SOURCE_TAG}_{quark_flavor}_{MOMENTUM_TAG}_{tmd_type}_{proj}_dt{tsep}_{gm}.h5"
                        )
                        combo_summary_file = (
                            combo_root / f"summary_{quark_flavor}_{tmd_type}_{proj}_dt{tsep}_{gm}.txt"
                        )
                        source_file_glob = (
                            f"*{tmd_type}*{quark_flavor}*dt{tsep}*{proj}.*.h5"
                        )
                        selected_path_globs = build_selected_path_globs(gm)

                        average_nucleon_tmdpdf_cfgs(
                            INPUT_ROOT,
                            combo_cfg_root,
                            cfg_dirs=CFG_LIST,
                            source_file_glob=source_file_glob,
                            source_file_suffix_token=gm,
                            selected_path_globs=selected_path_globs,
                            source_sink=SOURCE_SINK,
                            nt=Nt,
                            tsep=tsep,
                            apply_source_time_sign=APPLY_SOURCE_TIME_SIGN,
                            overwrite=OVERWRITE,
                            summary_file=combo_summary_file,
                        )
                        out = combine_tmdpdf_cfg_files(
                            combo_cfg_root,
                            combo_final_output,
                            source_sink=SOURCE_SINK,
                            overwrite=OVERWRITE,
                            summary_file=combo_summary_file,
                        )
                        print(
                            f"saved stripped tmdpdf file for {quark_flavor} {tmd_type} {proj} dt{tsep} {gm}: {out}"
                        )


if __name__ == "__main__":
    main()
