from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from latcorr.preprocess import discover_pion_softff_source_files, strip_pion_softff


def _make_softff_file(path: Path, value: complex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        root = f.create_group("srcZ5-X5_sinkZ5-X5")
        g = root.create_group("5_5")
        bx = g.create_group("bX_0")
        bx.create_dataset("ts10", data=np.full((64,), value, dtype=np.complex128))


def test_discover_and_strip_pion_softff(tmp_path):
    input_root = tmp_path / "input"
    file1 = input_root / "1050" / "HISQa076_140MeV.softFF.1050.ex_wall_PX0PY0PZ9_SP.x0y0z0t14.fw_qx0qy0qz4.bw_qx0qy0qz-5.h5"
    file2 = input_root / "1056" / "HISQa076_140MeV.softFF.1056.ex_wall_PX0PY0PZ9_SP.x0y0z0t22.fw_qx0qy0qz4.bw_qx0qy0qz-5.h5"
    _make_softff_file(file1, 1 + 1j)
    _make_softff_file(file2, 3 + 5j)

    files = discover_pion_softff_source_files(
        input_root,
        source_file_glob="*.fw_*.bw_*.h5",
    )
    assert len(files) == 2

    output_path = tmp_path / "out" / "softff.h5"
    summary_path = tmp_path / "out" / "summary.txt"
    out = strip_pion_softff(
        input_root,
        output_path,
        source_file_glob="*.fw_*.bw_*.h5",
        overwrite=True,
        summary_path=summary_path,
    )

    assert out == output_path
    with h5py.File(out, "r") as f:
        data = f["srcZ5-X5_sinkZ5-X5/5_5/bX_0/ts10"][...]
        np.testing.assert_allclose(data, np.full((64,), 2 + 3j, dtype=np.complex128))

    text = summary_path.read_text()
    assert "discovered 2 source files; used 2; failed 0; skipped_datasets 0" in text
