"""Pion soft form factor source-averaging helpers.

This workflow is intentionally simple because the softFF data volume is much
smaller than the nucleon three-point data:

1. discover source files recursively under the input root;
2. average each dataset path independently across all valid source files;
3. write one stripped ensemble HDF5 file.

The implementation keeps the HDF5 tree generic so it can tolerate partially
bad source files while still averaging the valid datasets.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


def _append_summary(summary_path: str | PathLike[str] | None, lines: Sequence[str]) -> None:
    """Append summary lines to ``summary_path`` if one was provided."""
    if summary_path is None or not lines:
        return

    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fout:
        for line in lines:
            fout.write(line.rstrip() + "\n")


def discover_pion_softff_source_files(
    input_root: str | PathLike[str],
    *,
    source_file_glob: str = "*.h5",
) -> list[Path]:
    """Return all softFF HDF5 source files under ``input_root``."""
    root = Path(input_root)
    return sorted(path for path in root.rglob(source_file_glob) if path.is_file())


def _flatten_hdf5_group(group: h5py.Group, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten datasets under ``group`` into a ``path -> array`` mapping."""
    out: dict[str, np.ndarray] = {}
    for key in group.keys():
        child = group[key]
        rel_path = f"{prefix}/{key}" if prefix else key
        if isinstance(child, h5py.Dataset):
            out[rel_path] = np.asarray(child)
        elif isinstance(child, h5py.Group):
            out.update(_flatten_hdf5_group(child, rel_path))
        else:
            raise TypeError(f"unsupported HDF5 node at {rel_path!r}: {type(child)!r}")
    return out


def _write_flat_hdf5(
    output_path: str | PathLike[str],
    datasets: dict[str, np.ndarray],
    *,
    root_group: str | None = None,
    overwrite: bool = True,
) -> Path:
    """Write a flattened dataset mapping back into an HDF5 tree."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"output file already exists: {out_path}")

    with h5py.File(out_path, "w") as fout:
        for rel_path in sorted(datasets):
            data = np.asarray(datasets[rel_path])
            grp = fout
            rel_parts = rel_path.split("/")
            if root_group is not None:
                grp = fout.require_group(root_group)
                rel_parts = rel_parts[1:] if rel_parts and rel_parts[0] == root_group else rel_parts
            for part in rel_parts[:-1]:
                grp = grp.require_group(part)
            dataset_name = rel_parts[-1]
            grp.create_dataset(dataset_name, data=data)

    return out_path


def read_source_pion_softff(
    source_file: str | PathLike[str],
) -> dict[str, np.ndarray]:
    """Read one source file into a flattened HDF5 tree.

    The entire file is flattened, so the original HDF5 path structure is
    preserved in the returned mapping keys.
    """
    source_path = Path(source_file)

    with h5py.File(source_path, "r") as h5f:
        return _flatten_hdf5_group(h5f)


def average_pion_softff_sources(
    source_files: Sequence[str | PathLike[str]],
    summary_path: str | PathLike[str] | None = None,
    summary_label: str = "pion softFF source files",
) -> dict[str, np.ndarray] | None:
    """Average all source files and return the resulting tree."""
    summary_lines: list[str] = []
    files = [Path(path) for path in source_files]
    if not files:
        summary_lines.append(f"{summary_label}: discovered 0 source files; used 0; failed 0; skipped_datasets 0")
        summary_lines.append("no source HDF5 files found")
        _append_summary(summary_path, summary_lines)
        return None

    accum: dict[str, np.ndarray] = {}
    expected_shapes: dict[str, tuple[int, ...]] = {}
    counts: dict[str, int] = {}
    used_count = 0
    failed_count = 0
    skipped_dataset_count = 0

    for source_file in files:
        try:
            tree = read_source_pion_softff(source_file)
        except Exception as exc:  # noqa: BLE001
            failed_count += 1
            summary_lines.append(f"skip source file {source_file}: {exc}")
            continue

        file_used = False
        file_had_skipped_dataset = False
        for path, arr in tree.items():
            if arr.ndim != 1:
                skipped_dataset_count += 1
                file_had_skipped_dataset = True
                summary_lines.append(
                    f"skip dataset {path} in {source_file}: expected 1D array, got shape {arr.shape}"
                )
                continue

            if np.isnan(arr).any():
                skipped_dataset_count += 1
                file_had_skipped_dataset = True
                summary_lines.append(f"skip dataset {path} in {source_file}: contains NaN")
                continue

            if path not in accum:
                accum[path] = np.array(arr, copy=True)
                expected_shapes[path] = arr.shape
                counts[path] = 1
            else:
                if arr.shape != expected_shapes[path]:
                    skipped_dataset_count += 1
                    file_had_skipped_dataset = True
                    summary_lines.append(
                        f"skip dataset {path} in {source_file}: shape mismatch"
                    )
                    continue
                accum[path] += arr
                counts[path] += 1

            file_used = True

        if file_used:
            used_count += 1
        if not file_used:
            failed_count += 1
            if not file_had_skipped_dataset:
                summary_lines.append(f"skip source file {source_file}: no valid datasets found")

    summary_lines.insert(
        0,
        (
            f"{summary_label}: discovered {len(files)} source files; "
            f"used {used_count}; failed {failed_count}; skipped_datasets {skipped_dataset_count}"
        ),
    )
    if not accum:
        _append_summary(summary_path, summary_lines)
        return None

    averaged = {path: values / float(counts[path]) for path, values in sorted(accum.items())}
    _append_summary(summary_path, summary_lines)
    return averaged


def strip_pion_softff(
    input_root: str | PathLike[str],
    output_path: str | PathLike[str],
    *,
    source_file_glob: str = "*.h5",
    overwrite: bool = True,
    summary_path: str | PathLike[str] | None = None,
) -> Path:
    """Run the full softFF preprocessing workflow."""
    if summary_path is not None and overwrite:
        summary_file = Path(summary_path)
        if summary_file.exists():
            summary_file.unlink()
    source_files = discover_pion_softff_source_files(input_root, source_file_glob=source_file_glob)
    summary_label = f"pion softFF source files matching {source_file_glob!r}"
    averaged = average_pion_softff_sources(
        source_files,
        summary_path=summary_path,
        summary_label=summary_label,
    )
    if averaged is None:
        raise ValueError(f"no valid source files found under {input_root!r}")
    return _write_flat_hdf5(output_path, averaged, root_group=None, overwrite=overwrite)
