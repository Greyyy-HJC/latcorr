"""Nucleon TMDPDF 3-point source-averaging helpers.

This module mirrors the two-step preprocessing style used for nucleon 2pt
data, but it is tuned for much larger 3pt TMDPDF files:

1. source files inside each configuration directory are read sequentially;
2. incomplete or unreadable files are skipped and logged to a summary file;
3. per-configuration averages are written to disk;
4. the per-configuration files are then combined into one ensemble file.

The implementation is intentionally generic over the HDF5 tree under the
``source_sink`` group, so it can tolerate the full nested qTMD layout used by
the legacy scripts.
"""

from __future__ import annotations

import re
from fnmatch import fnmatchcase
from os import PathLike
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

SOURCE_TIME_RE = re.compile(r"x(?P<x>-?\d+)y(?P<y>-?\d+)z(?P<z>-?\d+)t(?P<t>-?\d+)")


def discover_tmdpdf_source_files(
    cfg_dir: str | PathLike[str],
    *,
    source_file_glob: str = "*.h5",
    source_file_suffix_token: str | None = None,
) -> list[Path]:
    """Return all HDF5 source files inside one configuration directory.

    When ``source_file_suffix_token`` is provided, only files whose stem ends
    with ``.<token>`` are kept. This avoids glob overmatching when selecting
    files such as ``T`` that would otherwise also match ``SYT`` or ``SZT``.
    """
    cfg_path = Path(cfg_dir)
    files = [path for path in cfg_path.rglob(source_file_glob) if path.is_file()]
    if source_file_suffix_token is not None:
        files = [
            path
            for path in files
            if path.stem.split(".")[-1] == source_file_suffix_token
        ]
    return sorted(files)


def discover_tmdpdf_cfg_files(cfg_root: str | PathLike[str]) -> list[Path]:
    """Return per-configuration averaged files sorted by configuration number."""
    root = Path(cfg_root)
    cfg_files = [path for path in root.rglob("*.h5") if path.is_file() and path.stem.isdigit()]
    return sorted(cfg_files, key=lambda path: int(path.stem))


def parse_source_time(path: str | PathLike[str]) -> int:
    """Parse the source time coordinate from a source filename."""
    name = Path(path).name
    match = SOURCE_TIME_RE.search(name)
    if match is None:
        raise ValueError(f"cannot parse source position from filename: {name!r}")
    return int(match.group("t"))


def _flatten_hdf5_group(
    group: h5py.Group,
    prefix: str = "",
    *,
    include_globs: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Flatten datasets under ``group`` into a ``path -> array`` mapping.

    When ``include_globs`` is provided, only datasets whose relative path
    matches at least one glob pattern are materialized.
    """
    out: dict[str, np.ndarray] = {}
    for key in group.keys():
        child = group[key]
        rel_path = f"{prefix}/{key}" if prefix else key
        if isinstance(child, h5py.Dataset):
            if include_globs is not None and not any(fnmatchcase(rel_path, pat) for pat in include_globs):
                continue
            out[rel_path] = np.asarray(child)
        elif isinstance(child, h5py.Group):
            out.update(_flatten_hdf5_group(child, rel_path, include_globs=include_globs))
        else:
            raise TypeError(f"unsupported HDF5 node at {rel_path!r}: {type(child)!r}")
    return out


def _has_glob_wildcards(pattern: str) -> bool:
    """Return True when ``pattern`` contains glob wildcard characters."""
    return any(ch in pattern for ch in "*?[")


def _read_exact_hdf5_paths(
    group: h5py.Group,
    relative_paths: Sequence[str],
) -> dict[str, np.ndarray]:
    """Read a sequence of exact HDF5 dataset paths relative to ``group``.

    This is much faster than flattening the full tree when the requested paths
    are already known exactly.
    """
    out: dict[str, np.ndarray] = {}
    for rel_path in relative_paths:
        node = group
        for part in rel_path.split("/"):
            if not part:
                continue
            if part not in node:
                raise KeyError(f"missing dataset path component {part!r} in {rel_path!r}")
            node = node[part]
        if not isinstance(node, h5py.Dataset):
            raise KeyError(f"requested path is not a dataset: {rel_path!r}")
        out[rel_path] = np.asarray(node)
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


def _source_time_sign(
    source_file: str | PathLike[str],
    *,
    nt: int,
    tsep: int,
) -> float:
    """Return the legacy sign used for a given source file and source-sink separation."""
    source_t = parse_source_time(source_file)
    return -1.0 if source_t + tsep >= nt else 1.0


def _read_source_tree(
    source_file: str | PathLike[str],
    *,
    source_sink: str,
    apply_source_time_sign: bool,
    nt: int | None,
    tsep: int | None,
    selected_path_globs: Sequence[str] | None,
) -> dict[str, np.ndarray]:
    """Read one source file into a flattened HDF5 tree.

    When ``selected_path_globs`` is provided, only datasets whose relative
    HDF5 path matches one of the patterns are materialized.
    """
    source_path = Path(source_file)

    with h5py.File(source_path, "r") as h5f:
        if source_sink not in h5f:
            raise KeyError(f"missing source-sink group {source_sink!r} in {source_path}")
        root = h5f[source_sink]
        if selected_path_globs is None:
            data = _flatten_hdf5_group(root)
        else:
            exact_paths: list[str] = []
            glob_patterns: list[str] = []
            prefix = f"{source_sink}/"
            for pattern in selected_path_globs:
                rel_pattern = pattern[len(prefix):] if pattern.startswith(prefix) else pattern
                if _has_glob_wildcards(rel_pattern):
                    glob_patterns.append(rel_pattern)
                else:
                    exact_paths.append(rel_pattern)

            data = {}
            if exact_paths:
                data.update(_read_exact_hdf5_paths(root, exact_paths))
            if glob_patterns:
                data.update(_flatten_hdf5_group(root, include_globs=glob_patterns))

    if selected_path_globs is not None and not data:
        raise KeyError(
            f"no datasets matched the requested path filters in {source_path}"
        )

    if apply_source_time_sign:
        if nt is None or tsep is None:
            raise ValueError("nt and tsep are required when apply_source_time_sign=True")
        sign = _source_time_sign(source_path, nt=nt, tsep=tsep)
        data = {key: sign * value for key, value in data.items()}

    return data


def _average_source_trees(
    source_files: Sequence[str | PathLike[str]],
    *,
    source_sink: str,
    apply_source_time_sign: bool,
    nt: int | None,
    tsep: int | None,
    selected_path_globs: Sequence[str] | None,
    summary_label: str = "TMDPDF source files",
) -> tuple[dict[str, np.ndarray] | None, list[str]]:
    """Average valid source files and return the accumulated tree plus summary lines.

    ``selected_path_globs`` can be used to restrict source averaging to a
    chosen subset of HDF5 dataset paths.
    """
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
            tree = _read_source_tree(
                source_file,
                source_sink=source_sink,
                apply_source_time_sign=apply_source_time_sign,
                nt=nt,
                tsep=tsep,
                selected_path_globs=selected_path_globs,
            )
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

        assert accum is not None  # for type checkers

        shape_mismatch = False
        for key in expected_keys:
            if accum[key].shape != tree[key].shape:
                shape_mismatch = True
                break
        if shape_mismatch:
            failed_count += 1
            summary_lines.append(
                f"skip source file {source_file}: dataset shape mismatch"
            )
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

    averaged = {
        key: value / float(valid_count)
        for key, value in sorted(accum.items())
    }
    summary_lines.insert(
        0,
        (
            f"{summary_label}: discovered {len(files)} source files; "
            f"used {valid_count}; failed {failed_count}"
        ),
    )
    summary_lines.append(
        f"averaged {valid_count} valid source files into {len(averaged)} datasets"
    )
    return averaged, summary_lines


def process_tmdpdf_cfg_dir(
    cfg_dir: str | PathLike[str],
    output_path: str | PathLike[str],
    *,
    source_sink: str = "SS",
    source_file_glob: str = "*.h5",
    source_file_suffix_token: str | None = None,
    selected_path_globs: Sequence[str] | None = None,
    nt: int | None = None,
    tsep: int | None = None,
    apply_source_time_sign: bool = True,
    overwrite: bool = True,
    summary_path: str | PathLike[str] | None = None,
) -> Path | None:
    """Average all valid source files inside one configuration directory.

    The result is written to ``output_path`` and returned. If every source file
    in the configuration fails to read, ``None`` is returned and the failure is
    recorded in ``summary_path`` when provided.

    ``source_file_glob`` controls which files inside the configuration
    directory are considered part of the current gm / file selection.
    ``source_file_suffix_token`` can be used to require an exact suffix token
    such as ``T`` rather than matching every filename that merely ends with
    ``T.h5``.
    ``selected_path_globs`` controls which dataset paths inside the HDF5 tree
    are materialized before averaging.
    """
    cfg_path = Path(cfg_dir)
    summary_label = f"TMDPDF source files matching {source_file_glob!r} in cfg {cfg_path.name}"
    if source_file_suffix_token is not None:
        summary_label += f" with suffix token {source_file_suffix_token!r}"
    source_files = discover_tmdpdf_source_files(
        cfg_path,
        source_file_glob=source_file_glob,
        source_file_suffix_token=source_file_suffix_token,
    )
    averaged, summary_lines = _average_source_trees(
        source_files,
        source_sink=source_sink,
        apply_source_time_sign=apply_source_time_sign,
        nt=nt,
        tsep=tsep,
        selected_path_globs=selected_path_globs,
        summary_label=summary_label,
    )

    if averaged is None:
        summary_lines.insert(0, f"[cfg {cfg_path.name}] no output written")
        _append_summary(summary_path, summary_lines)
        return None

    out_path = _write_flat_hdf5(output_path, averaged, root_group=source_sink, overwrite=overwrite)
    summary_lines.insert(0, f"[cfg {cfg_path.name}] wrote {out_path}")
    _append_summary(summary_path, summary_lines)
    return out_path


def average_nucleon_tmdpdf_cfgs(
    input_root: str | PathLike[str],
    output_root: str | PathLike[str] | None = None,
    *,
    intermediate_root: str | PathLike[str] | None = None,
    cfg_dirs: Sequence[str | PathLike[str]],
    source_sink: str = "SS",
    source_file_glob: str = "*.h5",
    source_file_suffix_token: str | None = None,
    selected_path_globs: Sequence[str] | None = None,
    nt: int | None = None,
    tsep: int | None = None,
    apply_source_time_sign: bool = True,
    overwrite: bool = True,
    summary_path: str | PathLike[str] | None = None,
    summary_file: str | PathLike[str] | None = None,
) -> list[Path]:
    """Average the requested configuration directories under ``input_root``.

    ``cfg_dirs`` must be supplied explicitly.

    ``source_file_glob`` controls which files inside each configuration
    directory are considered part of the current gm / file selection.
    ``source_file_suffix_token`` can be used to require an exact suffix token
    such as ``T`` rather than matching every filename that merely ends with
    ``T.h5``.
    ``selected_path_globs`` controls which dataset paths inside the HDF5 tree
    are materialized before averaging.

    The per-configuration files are written as ``<output_root>/<cfg>.h5``.
    """
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
        result = process_tmdpdf_cfg_dir(
            cfg_path,
            cfg_output,
            source_sink=source_sink,
            source_file_glob=source_file_glob,
            source_file_suffix_token=source_file_suffix_token,
            selected_path_globs=selected_path_globs,
            nt=nt,
            tsep=tsep,
            apply_source_time_sign=apply_source_time_sign,
            overwrite=overwrite,
            summary_path=summary_path,
        )
        if result is not None:
            written.append(result)

    return written


def combine_tmdpdf_cfg_files(
    cfg_root: str | PathLike[str],
    output_path: str | PathLike[str],
    *,
    source_sink: str = "SS",
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
    cfg_files = discover_tmdpdf_cfg_files(cfg_root)
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
        summary_lines.append(
            f"no valid per-configuration files found under {cfg_root!r}"
        )
        _append_summary(summary_path, summary_lines)
        raise ValueError(f"no valid per-configuration files found under {cfg_root!r}")

    combined = {
        key: np.stack(per_path_arrays[key], axis=-1)
        for key in sorted(per_path_arrays)
    }
    out_path = _write_flat_hdf5(output_path, combined, root_group=source_sink, overwrite=overwrite)
    summary_lines.append(f"combined {valid_cfg_count} configuration files into {out_path}")
    _append_summary(summary_path, summary_lines)
    return out_path


def strip_nucleon_tmdpdf(
    input_root: str | PathLike[str],
    cfg_output_root: str | PathLike[str] | None = None,
    output_path: str | PathLike[str] | None = None,
    *,
    intermediate_root: str | PathLike[str] | None = None,
    final_output_path: str | PathLike[str] | None = None,
    cfg_dirs: Sequence[str | PathLike[str]],
    source_sink: str = "SS",
    source_file_glob: str = "*.h5",
    source_file_suffix_token: str | None = None,
    selected_path_globs: Sequence[str] | None = None,
    nt: int | None = None,
    tsep: int | None = None,
    apply_source_time_sign: bool = True,
    overwrite: bool = True,
    summary_path: str | PathLike[str] | None = None,
    summary_file: str | PathLike[str] | None = None,
) -> Path:
    """Run the full two-step TMDPDF preprocessing workflow.
    ``cfg_dirs`` must be supplied explicitly.

    ``source_file_glob`` controls which files inside each configuration
    directory are considered part of the current gm / file selection.
    ``source_file_suffix_token`` can be used to require an exact suffix token
    such as ``T`` rather than matching every filename that merely ends with
    ``T.h5``.
    ``selected_path_globs`` controls which dataset paths inside the HDF5 tree
    are materialized before averaging.
    """
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

    average_nucleon_tmdpdf_cfgs(
        input_root,
        cfg_output_root,
        cfg_dirs=cfg_dirs,
        source_sink=source_sink,
        source_file_glob=source_file_glob,
        source_file_suffix_token=source_file_suffix_token,
        selected_path_globs=selected_path_globs,
        nt=nt,
        tsep=tsep,
        apply_source_time_sign=apply_source_time_sign,
        overwrite=overwrite,
        summary_path=summary_path,
    )
    return combine_tmdpdf_cfg_files(
        cfg_output_root,
        output_path,
        source_sink=source_sink,
        overwrite=overwrite,
        summary_path=summary_path,
    )
