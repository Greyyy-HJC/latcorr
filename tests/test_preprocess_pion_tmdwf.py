from pathlib import Path

import h5py
import numpy as np

from latcorr.preprocess import (
    average_pion_tmdwf_cfgs,
    combine_pion_tmdwf_cfg_files,
    discover_pion_tmdwf_source_files,
    process_pion_tmdwf_cfg_dir,
    strip_pion_tmdwf,
)


def _write_pion_tmdwf_tree(path: Path, payload: np.ndarray, *, include_leaf: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as fout:
        grp = fout.require_group("SP")
        grp = grp.require_group("T5")
        grp = grp.require_group("PX0PY0PZ0")
        grp = grp.require_group("b_X")
        grp = grp.require_group("eta0")
        grp = grp.require_group("bT0")
        if include_leaf:
            grp.create_dataset("bz0", data=payload)


def test_discover_pion_tmdwf_source_files_filters_by_suffix(tmp_path):
    cfg_dir = tmp_path / "1050"
    cfg_dir.mkdir()
    _write_pion_tmdwf_tree(cfg_dir / "src_a.h5", np.arange(6.0))
    _write_pion_tmdwf_tree(cfg_dir / "src_b.dat", np.arange(6.0))

    files = discover_pion_tmdwf_source_files(cfg_dir, source_file_glob="*.h5")

    assert [path.name for path in files] == ["src_a.h5"]


def test_process_pion_tmdwf_cfg_dir_averages_tree_and_logs_skips(tmp_path):
    cfg_dir = tmp_path / "1050"
    cfg_dir.mkdir()
    _write_pion_tmdwf_tree(cfg_dir / "src_01.h5", np.array([1.0, 2.0, 3.0]))
    _write_pion_tmdwf_tree(cfg_dir / "src_09.h5", np.array([3.0, 6.0, 9.0]))
    _write_pion_tmdwf_tree(
        cfg_dir / "src_bad.h5",
        np.array([10.0, 20.0, 30.0]),
        include_leaf=False,
    )

    summary_path = tmp_path / "summary.txt"
    out_path = process_pion_tmdwf_cfg_dir(
        cfg_dir,
        tmp_path / "cfg" / "1050.h5",
        summary_path=summary_path,
    )

    assert out_path == tmp_path / "cfg" / "1050.h5"
    with h5py.File(out_path, "r") as h5f:
        data = np.asarray(h5f["SP/T5/PX0PY0PZ0/b_X/eta0/bT0/bz0"])
    np.testing.assert_allclose(data, np.array([2.0, 4.0, 6.0]))

    summary = summary_path.read_text()
    assert "pion TMDWF source files matching '*.h5' in cfg 1050: discovered 3 source files; used 2; failed 1" in summary
    assert "skip source file" in summary
    assert "averaged 2 valid source files" in summary


def test_average_and_combine_pion_tmdwf_cfgs(tmp_path):
    input_root = tmp_path / "input"
    cfg_1050 = input_root / "1050"
    cfg_1068 = input_root / "1068"
    _write_pion_tmdwf_tree(cfg_1050 / "src_01.h5", np.array([1.0, 2.0, 3.0]))
    _write_pion_tmdwf_tree(cfg_1050 / "src_09.h5", np.array([3.0, 6.0, 9.0]))
    _write_pion_tmdwf_tree(cfg_1068 / "src_00.h5", np.array([2.0, 4.0, 6.0]))

    cfg_root = tmp_path / "cfg"
    summary_path = tmp_path / "summary.txt"
    written = average_pion_tmdwf_cfgs(
        input_root,
        cfg_root,
        cfg_dirs=["1050", "1068"],
        summary_path=summary_path,
    )

    assert [path.name for path in written] == ["1050.h5", "1068.h5"]

    combined = combine_pion_tmdwf_cfg_files(
        cfg_root,
        tmp_path / "combined.h5",
        summary_path=summary_path,
    )

    with h5py.File(combined, "r") as h5f:
        data = np.asarray(h5f["SP/T5/PX0PY0PZ0/b_X/eta0/bT0/bz0"])

    np.testing.assert_allclose(
        data,
        np.array(
            [
                [2.0, 2.0],
                [4.0, 4.0],
                [6.0, 6.0],
            ]
        ),
    )

    summary = summary_path.read_text()
    assert "combined 2 configuration files" in summary


def test_strip_pion_tmdwf_runs_two_step_pipeline(tmp_path):
    input_root = tmp_path / "input"
    cfg_1050 = input_root / "1050"
    _write_pion_tmdwf_tree(cfg_1050 / "src_01.h5", np.array([1.0, 2.0, 3.0]))

    out = strip_pion_tmdwf(
        input_root,
        tmp_path / "cfg",
        tmp_path / "final.h5",
        cfg_dirs=["1050"],
        summary_path=tmp_path / "summary.txt",
    )

    with h5py.File(out, "r") as h5f:
        data = np.asarray(h5f["SP/T5/PX0PY0PZ0/b_X/eta0/bT0/bz0"])
    np.testing.assert_allclose(data, np.array([[1.0], [2.0], [3.0]]))
