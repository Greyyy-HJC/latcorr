from pathlib import Path

import h5py
import numpy as np

from latcorr.preprocess import (
    discover_pion_source_files,
    strip_pion_c2pt,
)


def _write_pion_source_tree(path: Path, payloads: dict[str, dict[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as fout:
        grp = fout.require_group("SS")
        for gamma, mom_map in payloads.items():
            grp_gamma = grp.require_group(gamma)
            for momentum, payload in mom_map.items():
                grp_gamma.create_dataset(momentum, data=payload)


def test_discover_pion_source_files_filters_by_suffix(tmp_path):
    cfg_dir = tmp_path / "1056"
    cfg_dir.mkdir()
    _write_pion_source_tree(
        cfg_dir / "HISQa076.c2pt.1056.ex.x39y43z45t7.1HYP_M140_GSRC_W45_k0.srcT5.h5",
        {"T": {"PX0PY0PZ0": np.arange(64.0)}},
    )
    _write_pion_source_tree(
        cfg_dir / "HISQa076.c2pt.1056.ex.x39y43z45t7.1HYP_M140_GSRC_W45_k0.srcX5.h5",
        {"T": {"PX0PY0PZ0": np.arange(64.0)}},
    )

    files = discover_pion_source_files(
        cfg_dir,
        source_type="srcT5",
        momentum_tag="1HYP_M140_GSRC_W45_k0",
    )

    assert [path.name for path in files] == [
        "HISQa076.c2pt.1056.ex.x39y43z45t7.1HYP_M140_GSRC_W45_k0.srcT5.h5"
    ]


def test_strip_pion_c2pt_runs_two_step_pipeline(tmp_path):
    input_root = tmp_path / "input"
    cfg_1056 = input_root / "1056"
    cfg_1062 = input_root / "1062"
    _write_pion_source_tree(
        cfg_1056 / "HISQa076.c2pt.1056.ex.x39y43z45t7.1HYP_M140_GSRC_W45_k0.srcT5.h5",
        {
            "T": {"PX0PY0PZ0": np.arange(64.0)},
            "X5": {"PX0PY0PZ0": np.arange(64.0) + 100.0},
        },
    )
    _write_pion_source_tree(
        cfg_1062 / "HISQa076.c2pt.1062.ex.x45y49z51t13.1HYP_M140_GSRC_W45_k0.srcT5.h5",
        {"T": {"PX0PY0PZ0": np.arange(64.0) + 4.0}},
    )

    out = strip_pion_c2pt(
        input_root,
        tmp_path / "final.h5",
        cfg_dirs=["1056", "1062"],
        source_type="srcT5",
        momentum_tag="1HYP_M140_GSRC_W45_k0",
        summary_path=tmp_path / "summary.txt",
    )

    with h5py.File(out, "r") as h5f:
        np.testing.assert_allclose(
            h5f["SS/T/PX0PY0PZ0"][()],
            np.column_stack([np.arange(64.0), np.arange(64.0) + 4.0]),
        )
        np.testing.assert_allclose(
            h5f["SS/X5/PX0PY0PZ0"][()],
            np.arange(64.0) + 100.0,
        )

    assert not (tmp_path / "cfg").exists()

    summary = (tmp_path / "summary.txt").read_text()
    assert "pion c2pt source files matching '*1HYP_M140_GSRC_W45_k0.srcT5.h5' in cfg 1056: discovered 1 source files; used 1; failed 0; skipped_datasets 0" in summary
