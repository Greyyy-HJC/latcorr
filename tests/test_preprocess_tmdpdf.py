from pathlib import Path

import h5py
import numpy as np

from latcorr.preprocess import (
    average_nucleon_tmdpdf_cfgs,
    combine_tmdpdf_cfg_files,
    parse_tmdpdf_source_time,
    process_tmdpdf_cfg_dir,
    strip_nucleon_tmdpdf,
)


def _write_source_tree(path: Path, payload: np.ndarray, *, include_leaf: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as fout:
        if include_leaf:
            grp = fout.require_group("SS")
            grp = grp.require_group("T")
            grp = grp.require_group("PX0PY0PZ0")
            grp = grp.require_group("b_X")
            grp = grp.require_group("eta0")
            grp = grp.require_group("bT0")
            grp.create_dataset("bz0", data=payload)


def test_parse_tmdpdf_source_time_from_filename():
    path = Path("qTMD_srcT5_1HYP_GSRC_W90_k0_PX0PY0PZ0.x33y37z23t17.h5")

    assert parse_tmdpdf_source_time(path) == 17


def test_process_tmdpdf_cfg_dir_skips_bad_source_files(tmp_path):
    cfg_dir = tmp_path / "1050"
    cfg_dir.mkdir()
    _write_source_tree(cfg_dir / "src.x00y00z00t01.h5", np.array([1.0, 2.0, 3.0]))
    _write_source_tree(cfg_dir / "src.x00y00z00t09.h5", np.array([3.0, 6.0, 9.0]))
    _write_source_tree(
        cfg_dir / "src.x00y00z00t04.h5",
        np.array([10.0, 20.0, 30.0]),
        include_leaf=False,
    )

    summary_path = tmp_path / "summary.txt"
    out_path = process_tmdpdf_cfg_dir(
        cfg_dir,
        tmp_path / "cfg" / "1050.h5",
        nt=12,
        tsep=4,
        summary_path=summary_path,
    )

    assert out_path == tmp_path / "cfg" / "1050.h5"
    with h5py.File(out_path, "r") as h5f:
        data = np.asarray(h5f["SS/T/PX0PY0PZ0/b_X/eta0/bT0/bz0"])
    np.testing.assert_allclose(data, np.array([-1.0, -2.0, -3.0]))

    summary = summary_path.read_text()
    assert "TMDPDF source files matching '*.h5' in cfg 1050: discovered 3 source files; used 2; failed 1" in summary
    assert "skip source file" in summary
    assert "averaged 2 valid source files" in summary


def test_process_tmdpdf_cfg_dir_filters_by_glob(tmp_path):
    cfg_dir = tmp_path / "1050"
    cfg_dir.mkdir()
    _write_source_tree(cfg_dir / "src_a_T.h5", np.array([1.0, 2.0, 3.0]))
    _write_source_tree(cfg_dir / "src_a_X.h5", np.array([9.0, 9.0, 9.0]))

    out_path = process_tmdpdf_cfg_dir(
        cfg_dir,
        tmp_path / "cfg" / "1050.h5",
        source_file_glob="*_T.h5",
        nt=12,
        tsep=4,
    )

    with h5py.File(out_path, "r") as h5f:
        data = np.asarray(h5f["SS/T/PX0PY0PZ0/b_X/eta0/bT0/bz0"])
    np.testing.assert_allclose(data, np.array([-1.0, -2.0, -3.0]))


def test_average_and_combine_tmdpdf_cfgs(tmp_path):
    input_root = tmp_path / "input"
    cfg_1050 = input_root / "1050"
    cfg_1068 = input_root / "1068"
    _write_source_tree(cfg_1050 / "src.x00y00z00t01.h5", np.array([1.0, 2.0, 3.0]))
    _write_source_tree(cfg_1050 / "src.x00y00z00t09.h5", np.array([3.0, 6.0, 9.0]))
    _write_source_tree(cfg_1068 / "src.x00y00z00t00.h5", np.array([2.0, 4.0, 6.0]))

    cfg_root = tmp_path / "cfg"
    summary_path = tmp_path / "summary.txt"
    written = average_nucleon_tmdpdf_cfgs(
        input_root,
        cfg_root,
        cfg_dirs=["1050", "1068"],
        nt=12,
        tsep=4,
        summary_path=summary_path,
    )

    assert [path.name for path in written] == ["1050.h5", "1068.h5"]

    combined = combine_tmdpdf_cfg_files(
        cfg_root,
        tmp_path / "combined.h5",
        summary_path=summary_path,
    )

    with h5py.File(combined, "r") as h5f:
        data = np.asarray(h5f["SS/T/PX0PY0PZ0/b_X/eta0/bT0/bz0"])

    np.testing.assert_allclose(
        data,
        np.array(
            [
                [-1.0, 2.0],
                [-2.0, 4.0],
                [-3.0, 6.0],
            ]
        ),
    )

    summary = summary_path.read_text()
    assert "TMDPDF source files matching '*.h5' in cfg 1050: discovered 3 source files; used 2; failed 1" in summary
    assert "combined 2 configuration files" in summary


def test_strip_nucleon_tmdpdf_runs_two_step_pipeline(tmp_path):
    input_root = tmp_path / "input"
    cfg_1050 = input_root / "1050"
    _write_source_tree(cfg_1050 / "src.x00y00z00t01.h5", np.array([1.0, 2.0, 3.0]))

    out = strip_nucleon_tmdpdf(
        input_root,
        tmp_path / "cfg",
        tmp_path / "final.h5",
        cfg_dirs=["1050"],
        nt=12,
        tsep=4,
        summary_path=tmp_path / "summary.txt",
    )

    with h5py.File(out, "r") as h5f:
        data = np.asarray(h5f["SS/T/PX0PY0PZ0/b_X/eta0/bT0/bz0"])
    np.testing.assert_allclose(data, np.array([[1.0], [2.0], [3.0]]))
