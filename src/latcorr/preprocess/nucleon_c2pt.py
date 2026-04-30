"""Nucleon 2pt source-averaging helpers.

This module is for the common pre-processing step where each configuration has
multiple source measurements that should be averaged before resampling or
downstream fitting.

The workflow is split into small steps:
1. load all source HDF5 files inside one configuration;
2. apply the source-time sign correction when requested;
3. average over sources inside the configuration;
4. stack all configurations and write the stripped ensemble file.
"""

from __future__ import annotations

import re
from os import PathLike
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

SOURCE_TIME_RE = re.compile(r"x(?P<x>-?\d+)y(?P<y>-?\d+)z(?P<z>-?\d+)t(?P<t>-?\d+)")


def _append_summary(summary_path: str | PathLike[str] | None, lines: Sequence[str]) -> None:
    """Append summary lines to ``summary_path`` if one was provided."""
    if summary_path is None or not lines:
        return

    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fout:
        for line in lines:
            fout.write(line.rstrip() + "\n")


def discover_source_files(
    cfg_dir: str | PathLike[str],
    *,
    source_file_glob: str = "*.h5",
) -> list[Path]:
    """Return HDF5 source files inside one configuration directory."""
    cfg_path = Path(cfg_dir)
    return sorted(path for path in cfg_path.rglob(source_file_glob) if path.is_file())


def parse_source_time(path: str | PathLike[str]) -> int:
    """Parse the source time coordinate from a source filename."""
    name = Path(path).name
    match = SOURCE_TIME_RE.search(name)
    if match is None:
        raise ValueError(f"cannot parse source position from filename: {name!r}")
    return int(match.group("t"))


def apply_antiperiodic_time_sign(data: np.ndarray, source_t: int, nt: int) -> np.ndarray:
    """Apply the (-1)^n wrap sign correction used in the legacy script."""
    arr = np.asarray(data)
    if arr.ndim != 1:
        raise ValueError(f"apply_antiperiodic_time_sign expects a 1D array, got {arr.shape}")
    if arr.shape[0] != nt:
        raise ValueError(f"array length {arr.shape[0]} does not match nt={nt}")

    t = np.arange(nt)
    n_wrap = (source_t + t) // nt
    sign = np.where(n_wrap % 2 == 0, 1.0, -1.0)
    return arr * sign


def read_source_c2pt(
    source_file: str | PathLike[str],
    *,
    gammas: Sequence[str],
    momenta: Sequence[str],
    source_sink: str = "SS",
    nt: int | None = None,
    apply_source_time_sign: bool = True,
) -> np.ndarray:
    """Read one source file into an array with shape ``(n_gamma, n_mom, nt)``."""
    source_path = Path(source_file)
    source_t = parse_source_time(source_path) if apply_source_time_sign else 0

    with h5py.File(source_path, "r") as h5f:
        gm_data: list[list[np.ndarray]] = []
        for gamma in gammas:
            mom_data: list[np.ndarray] = []
            for momentum in momenta:
                ds_path = f"{source_sink}/{gamma}/{momentum}"
                if ds_path not in h5f:
                    raise KeyError(f"missing dataset path: {ds_path}")

                arr = np.asarray(h5f[ds_path])
                if arr.ndim != 1:
                    raise ValueError(
                        f"expected a 1D correlator at {ds_path}, got shape {arr.shape}"
                    )
                if nt is None:
                    nt = arr.shape[0]
                elif arr.shape[0] != nt:
                    raise ValueError(
                        f"dataset length {arr.shape[0]} does not match expected nt={nt}"
                    )

                if apply_source_time_sign:
                    arr = apply_antiperiodic_time_sign(arr, source_t=source_t, nt=nt)

                mom_data.append(arr)
            gm_data.append(mom_data)

    return np.asarray(gm_data)


def average_source_files(
    source_files: Sequence[str | PathLike[str]],
    *,
    gammas: Sequence[str],
    momenta: Sequence[str],
    source_sink: str = "SS",
    nt: int | None = None,
    apply_source_time_sign: bool = True,
    summary_path: str | PathLike[str] | None = None,
    summary_label: str = "2pt source HDF5 files",
) -> np.ndarray:
    """Average all source files inside one configuration directory."""
    summary_lines: list[str] = []
    files = [Path(path) for path in source_files]
    accum: np.ndarray | None = None
    expected_shape: tuple[int, ...] | None = None
    used_count = 0
    failed_count = 0

    for source_file in files:
        try:
            arr = read_source_c2pt(
                source_file,
                gammas=gammas,
                momenta=momenta,
                source_sink=source_sink,
                nt=nt,
                apply_source_time_sign=apply_source_time_sign,
            )
        except Exception as exc:  # noqa: BLE001
            failed_count += 1
            summary_lines.append(f"skip source file {source_file}: {exc}")
            continue

        if accum is None:
            accum = np.array(arr, copy=True)
            expected_shape = accum.shape
        else:
            if arr.shape != expected_shape:
                failed_count += 1
                summary_lines.append(
                    f"skip source file {source_file}: array shape mismatch"
                )
                continue
            accum += arr
        used_count += 1

    summary_lines.insert(
        0,
        (
            f"{summary_label}: discovered {len(files)} source files; "
            f"used {used_count}; failed {failed_count}"
        ),
    )
    if accum is None or used_count == 0:
        _append_summary(summary_path, summary_lines)
        raise ValueError("no source HDF5 files found for the configuration")

    _append_summary(summary_path, summary_lines)
    return accum / float(used_count)


def process_cfg_dir(
    cfg_dir: str | PathLike[str],
    *,
    gammas: Sequence[str],
    momenta: Sequence[str],
    source_sink: str = "SS",
    source_file_glob: str = "*.h5",
    nt: int | None = None,
    apply_source_time_sign: bool = True,
    summary_path: str | PathLike[str] | None = None,
) -> np.ndarray:
    """Process one configuration directory and return the source-averaged data."""
    source_files = discover_source_files(cfg_dir, source_file_glob=source_file_glob)
    summary_label = f"2pt source HDF5 files in cfg {Path(cfg_dir).name}"
    if not source_files:
        _append_summary(
            summary_path,
            [f"{summary_label}: discovered 0 source files; used 0; failed 0"],
        )
        raise ValueError(f"no HDF5 source files found in cfg directory: {cfg_dir!r}")
    return average_source_files(
        source_files,
        gammas=gammas,
        momenta=momenta,
        source_sink=source_sink,
        nt=nt,
        apply_source_time_sign=apply_source_time_sign,
        summary_path=summary_path,
        summary_label=summary_label,
    )


def build_ensemble_c2pt(
    cfg_dirs: Sequence[str | PathLike[str]],
    *,
    gammas: Sequence[str],
    momenta: Sequence[str],
    source_sink: str = "SS",
    source_file_glob: str = "*.h5",
    nt: int | None = None,
    apply_source_time_sign: bool = True,
    summary_path: str | PathLike[str] | None = None,
) -> np.ndarray:
    """Build the stacked ensemble array with shape ``(n_cfg, n_gamma, n_mom, nt)``."""
    cfg_arrays = [
        process_cfg_dir(
            cfg_dir,
            gammas=gammas,
            momenta=momenta,
            source_sink=source_sink,
            source_file_glob=source_file_glob,
            nt=nt,
            apply_source_time_sign=apply_source_time_sign,
            summary_path=summary_path,
        )
        for cfg_dir in cfg_dirs
    ]
    if not cfg_arrays:
        raise ValueError("cfg_dirs is empty")
    return np.stack(cfg_arrays, axis=0)


def write_stripped_c2pt(
    output_path: str | PathLike[str],
    ensemble_data: np.ndarray,
    *,
    gammas: Sequence[str],
    momenta: Sequence[str],
    source_sink: str = "SS",
    overwrite: bool = True,
) -> Path:
    """Write the stripped ensemble to an HDF5 file.

    The output layout matches the existing reader expectations:
    each dataset is stored as ``(nt, n_cfg)`` under ``source_sink/gamma/momentum``.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"output file already exists: {out_path}")

    data = np.asarray(ensemble_data)
    if data.ndim != 4:
        raise ValueError(
            "ensemble_data must have shape (n_cfg, n_gamma, n_mom, nt), "
            f"got {data.shape}"
        )

    if data.shape[1] != len(gammas) or data.shape[2] != len(momenta):
        raise ValueError(
            "ensemble_data shape does not match the provided gamma/momentum lists"
        )

    with h5py.File(out_path, "w") as fout:
        grp_source = fout.require_group(source_sink)
        for gamma_idx, gamma in enumerate(gammas):
            grp_gamma = grp_source.require_group(gamma)
            for mom_idx, momentum in enumerate(momenta):
                dataset = np.transpose(data[:, gamma_idx, mom_idx, :], (1, 0))
                grp_gamma.create_dataset(momentum, data=dataset)

    return out_path


def strip_nucleon_c2pt(
    input_root: str | PathLike[str],
    output_path: str | PathLike[str],
    *,
    gammas: Sequence[str],
    momenta: Sequence[str],
    source_sink: str = "SS",
    source_file_glob: str = "*.h5",
    cfg_dirs: Sequence[str | PathLike[str]],
    nt: int | None = None,
    apply_source_time_sign: bool = True,
    overwrite: bool = True,
    summary_path: str | PathLike[str] | None = None,
) -> Path:
    """Convenience wrapper that performs the full source-averaging workflow."""
    ensemble_data = build_ensemble_c2pt(
        cfg_dirs,
        gammas=gammas,
        momenta=momenta,
        source_sink=source_sink,
        source_file_glob=source_file_glob,
        nt=nt,
        apply_source_time_sign=apply_source_time_sign,
        summary_path=summary_path,
    )
    return write_stripped_c2pt(
        output_path,
        ensemble_data,
        gammas=gammas,
        momenta=momenta,
        source_sink=source_sink,
        overwrite=overwrite,
    )
