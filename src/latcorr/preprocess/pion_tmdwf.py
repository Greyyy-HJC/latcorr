"""Pion TMDWF source-averaging helpers.

This workflow matches the two-step style used by the nucleon TMDPDF
preprocessing:

1. source files inside each configuration directory are read sequentially;
2. unreadable or incomplete files are skipped and logged to a summary file;
3. per-configuration averages are written to disk;
4. the per-configuration files are then combined into one ensemble file.

The implementation keeps the HDF5 tree generic under ``source_sink`` so it can
handle the nested qTMDWF layout used by the legacy split/combine scripts.
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


def _coalesce_alias(
    primary: str | PathLike[str] | None,
    alias: str | PathLike[str] | None,
    *,
    primary_name: str,
    alias_name: str,
) -> str | PathLike[str]:
    """Return whichever of two equivalent arguments was supplied."""
    if primary is not None and alias is not None and Path(primary) != Path(alias):
        raise ValueError(f"{primary_name} and {alias_name} refer to different paths")
    if primary is not None:
        return primary
    if alias is not None:
        return alias
    raise ValueError(f"one of {primary_name} or {alias_name} must be provided")


def _coalesce_optional_alias(
    primary: str | PathLike[str] | None,
    alias: str | PathLike[str] | None,
    *,
    primary_name: str,
    alias_name: str,
) -> str | PathLike[str] | None:
    """Return an optional argument, allowing both forms to be omitted."""
    if primary is not None and alias is not None and Path(primary) != Path(alias):
        raise ValueError(f"{primary_name} and {alias_name} refer to different paths")
    if primary is not None:
        return primary
    return alias


def discover_pion_tmdwf_source_files(
    cfg_dir: str | PathLike[str],
    *,
    source_file_glob: str = "*.h5",
) -> list[Path]:
    """Return all HDF5 source files inside one configuration directory."""
    cfg_path = Path(cfg_dir)
    return sorted(path for path in cfg_path.rglob(source_file_glob) if path.is_file())


def discover_pion_tmdwf_cfg_files(cfg_root: str | PathLike[str]) -> list[Path]:
    """Return per-configuration averaged files sorted by configuration number."""
    root = Path(cfg_root)
    cfg_files = [path for path in root.rglob("*.h5") if path.is_file() and path.stem.isdigit()]
    return sorted(cfg_files, key=lambda path: int(path.stem))


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


def read_source_pion_tmdwf(
    source_file: str | PathLike[str],
    *,
    source_sink: str = "SP",
) -> dict[str, np.ndarray]:
    """Read one source file into a flattened HDF5 tree."""
    source_path = Path(source_file)

    with h5py.File(source_path, "r") as h5f:
        if source_sink not in h5f:
            raise KeyError(f"missing source-sink group {source_sink!r} in {source_path}")
        return _flatten_hdf5_group(h5f[source_sink])


def _average_source_trees(
    source_files: Sequence[str | PathLike[str]],
    *,
    source_sink: str,
    summary_label: str = "pion TMDWF source files",
) -> tuple[dict[str, np.ndarray] | None, list[str]]:
    """Average valid source files and return the accumulated tree plus summary lines."""
    summary_lines: list[str] = []
    files = [Path(path) for path in source_files]
    if not files:
        summary_lines.append(f"{summary_label}: discovered 0 source files; used 0; failed 0")
        summary_lines.append("no source HDF5 files found")
        return None, summary_lines

    accum: dict[str, np.ndarray] | None = None
    expected_keys: set[str] | None = None
    valid_count = 0
    failed_count = 0

    for source_file in files:
        try:
            tree = read_source_pion_tmdwf(source_file, source_sink=source_sink)
        except Exception as exc:  # noqa: BLE001
            failed_count += 1
            summary_lines.append(f"skip source file {source_file}: {exc}")
            continue

        tree_keys = set(tree)
        if expected_keys is None:
            expected_keys = tree_keys
            accum = {key: np.array(value, copy=True) for key, value in tree.items()}
            valid_count = 1
            continue

        if tree_keys != expected_keys:
            failed_count += 1
            summary_lines.append(
                "skip source file "
                f"{source_file}: dataset tree mismatch with the first valid file"
            )
            continue

        assert accum is not None

        shape_mismatch = False
        for key in expected_keys:
            if accum[key].shape != tree[key].shape:
                shape_mismatch = True
                break
        if shape_mismatch:
            failed_count += 1
            summary_lines.append(f"skip source file {source_file}: dataset shape mismatch")
            continue

        for key in expected_keys:
            accum[key] += tree[key]
        valid_count += 1

    if accum is None or valid_count == 0:
        summary_lines.insert(
            0,
            (
                f"{summary_label}: discovered {len(files)} source files; "
                f"used {valid_count}; failed {failed_count}"
            ),
        )
        summary_lines.append("no valid source files were found for this configuration")
        return None, summary_lines

    summary_lines.insert(
        0,
        (
            f"{summary_label}: discovered {len(files)} source files; "
            f"used {valid_count}; failed {failed_count}"
        ),
    )
    summary_lines.append(f"averaged {valid_count} valid source files")
    averaged = {key: values / float(valid_count) for key, values in sorted(accum.items())}
    return averaged, summary_lines


def process_pion_tmdwf_cfg_dir(
    cfg_dir: str | PathLike[str],
    output_path: str | PathLike[str],
    *,
    source_sink: str = "SP",
    source_file_glob: str = "*.h5",
    overwrite: bool = True,
    summary_path: str | PathLike[str] | None = None,
) -> Path | None:
    """Average all valid source files inside one configuration directory."""
    cfg_path = Path(cfg_dir)
    summary_label = f"pion TMDWF source files matching {source_file_glob!r} in cfg {cfg_path.name}"
    source_files = discover_pion_tmdwf_source_files(cfg_path, source_file_glob=source_file_glob)
    averaged, summary_lines = _average_source_trees(
        source_files,
        source_sink=source_sink,
        summary_label=summary_label,
    )
    _append_summary(summary_path, summary_lines)

    if averaged is None:
        _append_summary(summary_path, [f"[cfg {cfg_path.name}] no output written"])
        return None

    out_path = _write_flat_hdf5(output_path, averaged, root_group=source_sink, overwrite=overwrite)
    _append_summary(summary_path, [f"[cfg {cfg_path.name}] wrote {out_path}"])
    return out_path


def average_pion_tmdwf_cfgs(
    input_root: str | PathLike[str],
    output_root: str | PathLike[str] | None = None,
    *,
    intermediate_root: str | PathLike[str] | None = None,
    cfg_dirs: Sequence[str | PathLike[str]],
    source_sink: str = "SP",
    source_file_glob: str = "*.h5",
    overwrite: bool = True,
    summary_path: str | PathLike[str] | None = None,
    summary_file: str | PathLike[str] | None = None,
) -> list[Path]:
    """Average the requested configuration directories under ``input_root``."""
    output_root = _coalesce_alias(
        output_root,
        intermediate_root,
        primary_name="output_root",
        alias_name="intermediate_root",
    )
    summary_path = _coalesce_optional_alias(
        summary_path,
        summary_file,
        primary_name="summary_path",
        alias_name="summary_file",
    )

    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for cfg_dir in cfg_dirs:
        cfg_path = Path(cfg_dir)
        if not cfg_path.is_absolute():
            cfg_path = Path(input_root) / cfg_path
        cfg_output = output_path / f"{cfg_path.name}.h5"
        result = process_pion_tmdwf_cfg_dir(
            cfg_path,
            cfg_output,
            source_sink=source_sink,
            source_file_glob=source_file_glob,
            overwrite=overwrite,
            summary_path=summary_path,
        )
        if result is not None:
            written.append(result)

    return written


def combine_pion_tmdwf_cfg_files(
    cfg_root: str | PathLike[str],
    output_path: str | PathLike[str],
    *,
    source_sink: str = "SP",
    overwrite: bool = True,
    summary_path: str | PathLike[str] | None = None,
    summary_file: str | PathLike[str] | None = None,
) -> Path:
    """Combine per-configuration averaged files into one ensemble file."""
    summary_path = _coalesce_optional_alias(
        summary_path,
        summary_file,
        primary_name="summary_path",
        alias_name="summary_file",
    )

    cfg_files = discover_pion_tmdwf_cfg_files(cfg_root)
    summary_lines: list[str] = []
    if not cfg_files:
        raise ValueError(f"no per-configuration HDF5 files found under {cfg_root!r}")

    per_path_arrays: dict[str, list[np.ndarray]] = {}
    expected_keys: set[str] | None = None
    valid_cfg_count = 0

    for cfg_file in cfg_files:
        try:
            with h5py.File(cfg_file, "r") as h5f:
                if source_sink not in h5f:
                    raise KeyError(f"missing source-sink group {source_sink!r}")
                tree = _flatten_hdf5_group(h5f[source_sink])
        except Exception as exc:  # noqa: BLE001
            summary_lines.append(f"skip cfg file {cfg_file}: {exc}")
            continue

        tree_keys = set(tree)
        if expected_keys is None:
            expected_keys = tree_keys
            per_path_arrays = {key: [value] for key, value in tree.items()}
            valid_cfg_count = 1
            continue

        if tree_keys != expected_keys:
            summary_lines.append(
                f"skip cfg file {cfg_file}: dataset tree mismatch with the first valid file"
            )
            continue

        shape_mismatch = False
        for key in expected_keys:
            if per_path_arrays[key][0].shape != tree[key].shape:
                shape_mismatch = True
                break
        if shape_mismatch:
            summary_lines.append(f"skip cfg file {cfg_file}: dataset shape mismatch")
            continue

        for key in expected_keys:
            per_path_arrays[key].append(tree[key])
        valid_cfg_count += 1

    if expected_keys is None or not per_path_arrays:
        summary_lines.append(f"no valid per-configuration files found under {cfg_root!r}")
        _append_summary(summary_path, summary_lines)
        raise ValueError(f"no valid per-configuration files found under {cfg_root!r}")

    combined = {key: np.stack(per_path_arrays[key], axis=-1) for key in sorted(per_path_arrays)}
    out_path = _write_flat_hdf5(output_path, combined, root_group=source_sink, overwrite=overwrite)
    summary_lines.append(f"combined {valid_cfg_count} configuration files into {out_path}")
    _append_summary(summary_path, summary_lines)
    return out_path


def strip_pion_tmdwf(
    input_root: str | PathLike[str],
    cfg_output_root: str | PathLike[str] | None = None,
    output_path: str | PathLike[str] | None = None,
    *,
    intermediate_root: str | PathLike[str] | None = None,
    final_output_path: str | PathLike[str] | None = None,
    cfg_dirs: Sequence[str | PathLike[str]],
    source_sink: str = "SP",
    source_file_glob: str = "*.h5",
    overwrite: bool = True,
    summary_path: str | PathLike[str] | None = None,
    summary_file: str | PathLike[str] | None = None,
) -> Path:
    """Run the full two-step pion TMDWF preprocessing workflow."""
    cfg_output_root = _coalesce_alias(
        cfg_output_root,
        intermediate_root,
        primary_name="cfg_output_root",
        alias_name="intermediate_root",
    )
    output_path = _coalesce_alias(
        output_path,
        final_output_path,
        primary_name="output_path",
        alias_name="final_output_path",
    )
    summary_path = _coalesce_optional_alias(
        summary_path,
        summary_file,
        primary_name="summary_path",
        alias_name="summary_file",
    )

    average_pion_tmdwf_cfgs(
        input_root,
        cfg_output_root,
        cfg_dirs=cfg_dirs,
        source_sink=source_sink,
        source_file_glob=source_file_glob,
        overwrite=overwrite,
        summary_path=summary_path,
    )
    return combine_pion_tmdwf_cfg_files(
        cfg_output_root,
        output_path,
        source_sink=source_sink,
        overwrite=overwrite,
        summary_path=summary_path,
    )
