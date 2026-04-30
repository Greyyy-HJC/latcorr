from pathlib import Path

import h5py
import numpy as np

from latcorr.preprocess import (
    apply_antiperiodic_time_sign,
    parse_source_time,
    strip_nucleon_c2pt,
)


def test_parse_source_time_from_filename():
    path = Path(
        "l64c64a076.c2pt.1050.ex.x33y37z23t17.1HYP_GSRC_W90_k0_5.h5"
    )

    assert parse_source_time(path) == 17


def test_apply_antiperiodic_time_sign_matches_legacy_rule():
    data = np.arange(8.0)

    out = apply_antiperiodic_time_sign(data, source_t=1, nt=8)

    np.testing.assert_allclose(out, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -7.0]))


def test_strip_nucleon_c2pt_uses_explicit_cfg_dirs(tmp_path):
    input_root = tmp_path / "input"
    for cfg in ["1050", "1068"]:
        cfg_dir = input_root / cfg
        cfg_dir.mkdir(parents=True)
        with h5py.File(cfg_dir / f"src.x00y00z00t0{cfg[-1]}.h5", "w") as fout:
            grp = fout.require_group("SS")
            grp = grp.require_group("T")
            grp = grp.require_group("PX0PY0PZ0")
            grp.create_dataset("d0", data=np.arange(8.0))
    with h5py.File(input_root / "1050" / "src.x00y00z00t99.h5", "w") as fout:
        fout.require_group("SS")

    summary_path = tmp_path / "summary.txt"
    out = strip_nucleon_c2pt(
        input_root,
        tmp_path / "out.h5",
        cfg_dirs=["1050", "1068"],
        gammas=["T"],
        momenta=["d0"],
        nt=8,
        summary_path=summary_path,
    )

    assert out == tmp_path / "out.h5"
    with h5py.File(out, "r") as fout:
        assert np.asarray(fout["SS/T/d0"]).shape == (8, 2)

    summary = summary_path.read_text()
    assert "2pt source HDF5 files in cfg 1050: discovered 2 source files; used 1; failed 1" in summary
    assert "2pt source HDF5 files in cfg 1068: discovered 1 source files; used 1; failed 0" in summary
