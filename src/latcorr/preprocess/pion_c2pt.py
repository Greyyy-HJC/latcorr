"""Pion 2pt source-averaging helpers for qTMDWF data.

The workflow mirrors the nucleon 2pt preprocessing style:
1. discover source files inside one configuration directory;
2. average source measurements inside each configuration;
3. stack the configuration averages and write one stripped ensemble file.

The legacy qTMDWF files store many correlator datasets under ``SS``. We keep
the tree generic and average each dataset path independently so that partially
bad source files can still contribute their valid datasets.
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


def discover_pion_source_files(
    cfg_dir: str | PathLike[str],
    *,
    source_type: str,
    momentum_tag: str,
    source_file_glob: str | None = None,
) -> list[Path]:
    """Return all HDF5 source files inside one configuration directory."""
    cfg_path = Path(cfg_dir)
    glob_pattern = source_file_glob or f"*{momentum_tag}.{source_type}.h5"
    return sorted(path for path in cfg_path.rglob(glob_pattern) if path.is_file())


def _flatten_hdf5_group(group: h5py.Group, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten every dataset under ``group`` into a ``path -> array`` mapping."""
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
    root_group: str,
    overwrite: bool = True,
) -> Path:
    """Write a flattened dataset mapping back into an HDF5 tree."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"output file already exists: {out_path}")

    with h5py.File(out_path, "w") as fout:
        root = fout.require_group(root_group)
        for rel_path in sorted(datasets):
            data = np.asarray(datasets[rel_path])
            if "/" in rel_path:
                group_path, dataset_name = rel_path.rsplit("/", 1)
                grp = root
                for part in group_path.split("/"):
                    grp = grp.require_group(part)
            else:
                grp = root
                dataset_name = rel_path
            grp.create_dataset(dataset_name, data=data)

    return out_path


def read_source_pion_c2pt(
    source_file: str | PathLike[str],
    *,
    source_sink: str = "SS",
) -> dict[str, np.ndarray]:
    """Read one source file into a flattened HDF5 tree."""
    source_path = Path(source_file)

    with h5py.File(source_path, "r") as h5f:
        if source_sink not in h5f:
            raise KeyError(f"missing source-sink group {source_sink!r} in {source_path}")
        return _flatten_hdf5_group(h5f[source_sink])


def average_source_files(
    source_files: Sequence[str | PathLike[str]],
    *,
    source_sink: str = "SS",
    summary_path: str | PathLike[str] | None = None,
    summary_label: str = "pion c2pt source files",
) -> dict[str, np.ndarray] | None:
    """Average all source files inside one configuration directory.

    Each dataset path is averaged independently so that one bad dataset does
    not invalidate the full source file.
    """
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
            tree = read_source_pion_c2pt(source_file, source_sink=source_sink)
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


def _combine_pion_cfg_trees(
    cfg_trees: Sequence[dict[str, np.ndarray]],
    cfg_names: Sequence[str],
    output_path: str | PathLike[str],
    *,
    source_sink: str = "SS",
    overwrite: bool = True,
    summary_path: str | PathLike[str] | None = None,
) -> Path:
    """Combine in-memory per-configuration trees into one ensemble file."""
    summary_lines: list[str] = []
    if not cfg_trees:
        summary_lines.append("no valid per-configuration files found")
        _append_summary(summary_path, summary_lines)
        raise ValueError("no valid per-configuration files found")

    per_path_arrays: dict[str, list[np.ndarray]] = {}
    per_path_cfgs: dict[str, list[str]] = {}
    per_path_shapes: dict[str, tuple[int, ...]] = {}
    valid_cfg_count = 0

    for cfg_name, tree in zip(cfg_names, cfg_trees):
        cfg_used = False
        for key, arr in tree.items():
            if key not in per_path_shapes:
                per_path_shapes[key] = arr.shape
            elif arr.shape != per_path_shapes[key]:
                summary_lines.append(
                    f"skip dataset {key} in cfg file {cfg_name}: shape mismatch"
                )
                continue

            per_path_arrays.setdefault(key, []).append(arr)
            per_path_cfgs.setdefault(key, []).append(cfg_name)
            cfg_used = True

        if cfg_used:
            valid_cfg_count += 1
        else:
            summary_lines.append(f"skip cfg file {cfg_name}: no valid datasets found")

    if not per_path_arrays:
        summary_lines.append("no valid per-configuration files found")
        _append_summary(summary_path, summary_lines)
        raise ValueError("no valid per-configuration files found")

    combined = {
        key: np.stack(per_path_arrays[key], axis=-1)
        for key in sorted(per_path_arrays)
    }
    out_path = _write_flat_hdf5(output_path, combined, root_group=source_sink, overwrite=overwrite)

    with h5py.File(out_path, "a") as fout:
        root = fout.require_group(source_sink)
        for key, cfg_list in per_path_cfgs.items():
            if "/" in key:
                group_path, dataset_name = key.rsplit("/", 1)
                grp = root
                for part in group_path.split("/"):
                    grp = grp.require_group(part)
            else:
                grp = root
                dataset_name = key
            grp[dataset_name].attrs["configs"] = [np.bytes_(cfg) for cfg in cfg_list]

    summary_lines.append(f"combined {valid_cfg_count} configuration files into {out_path}")
    _append_summary(summary_path, summary_lines)
    return out_path


def strip_pion_c2pt(
    input_root: str | PathLike[str],
    output_path: str | PathLike[str] | None = None,
    *,
    cfg_dirs: Sequence[str | PathLike[str]],
    source_type: str,
    momentum_tag: str,
    source_sink: str = "SS",
    source_file_glob: str | None = None,
    overwrite: bool = True,
    summary_path: str | PathLike[str] | None = None,
    summary_file: str | PathLike[str] | None = None,
) -> Path:
    """Run the full pion c2pt preprocessing workflow without cfg intermediates."""
    if output_path is None:
        raise ValueError("output_path must be provided")
    if summary_path is not None and summary_file is not None and Path(summary_path) != Path(summary_file):
        raise ValueError("summary_path and summary_file refer to different paths")
    if summary_path is None:
        summary_path = summary_file

    cfg_trees: list[dict[str, np.ndarray]] = []
    cfg_names: list[str] = []

    for cfg_dir in cfg_dirs:
        cfg_path = Path(cfg_dir)
        if not cfg_path.is_absolute():
            cfg_path = Path(input_root) / cfg_path
        summary_label = (
            f"pion c2pt source files matching {source_file_glob or f'*{momentum_tag}.{source_type}.h5'!r} "
            f"in cfg {cfg_path.name}"
        )
        source_files = discover_pion_source_files(
            cfg_path,
            source_type=source_type,
            momentum_tag=momentum_tag,
            source_file_glob=source_file_glob,
        )
        averaged = average_source_files(
            source_files,
            source_sink=source_sink,
            summary_path=summary_path,
            summary_label=summary_label,
        )
        if averaged is None:
            _append_summary(summary_path, [f"[cfg {cfg_path.name}] no output written"])
            continue
        cfg_trees.append(averaged)
        cfg_names.append(cfg_path.name)

    return _combine_pion_cfg_trees(
        cfg_trees,
        cfg_names,
        output_path,
        source_sink=source_sink,
        overwrite=overwrite,
        summary_path=summary_path,
    )
