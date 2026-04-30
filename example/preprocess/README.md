# Preprocess Examples

This folder contains small, runnable examples for `latcorr.preprocess`.
`basic_preprocess.py` is the single entry point for the synthetic basic demo
and the synthetic nucleon TMDPDF demo.

## `basic_preprocess.py`

Single entry point for two synthetic demos:

- basic correlator cleanup, symmetrization, and normalization;
- nucleon TMDPDF-style source averaging and correlator preprocessing on
  synthetic arrays.

The parameter blocks at the top of the script let you tweak both demos without
touching the helper code.

## `strip_nucleon_c2pt.py`

Strip nucleon 2pt source averages from per-configuration HDF5 folders.

### What it does

- Takes an explicit `CFG_LIST`.
- Builds per-cfg input directories from `INPUT_ROOT / cfg`.
- Reads all source HDF5 files inside each cfg directory.
- Uses the explicit `SOURCE_FILE_GLOB` pattern to make the file selection clear.
- Averages source measurements inside each cfg.
- Applies the source-time sign flip that comes from the antiperiodic
  temporal boundary condition when `APPLY_SOURCE_TIME_SIGN` is enabled.
- Stacks the cfg averages and writes one stripped ensemble file.
- Writes a summary file that records:
  - how many source files were discovered
  - how many were used
  - how many failed

### How to use it

- Edit the parameter block at the top of the script.
- Set `INPUT_ROOT` to the directory that contains the cfg folders.
- Set `OUTPUT_PATH` to the final HDF5 file location.
- Set `SUMMARY_PATH` if you want a separate summary text file.
- Set `ENSEMBLE_TAG` to match the ensemble name used in the input path.
- Set `CFG_LIST` to the cfg IDs you want to process.
- Run the script directly with Python.

### Main knobs

- `srctype`
- `ENSEMBLE_TAG`
- `MOMENTUM_TAG`
- `CFG_LIST`
- `INPUT_ROOT`
- `OUTPUT_PATH`
- `SUMMARY_PATH`
- `GAMMAS`
- `MOMENTA`
- `SOURCE_SINK`
- `SOURCE_FILE_GLOB`
- `NT`
- `APPLY_SOURCE_TIME_SIGN`

### Example command

```bash
python3.11 example/preprocess/strip_nucleon_c2pt.py
```

## `strip_pion_c2pt.py`

Strip pion 2pt source averages from qTMDWF per-configuration HDF5 folders.

### What it does

- Takes an explicit `CFG_LIST`.
- Builds per-cfg input directories from `INPUT_ROOT / cfg`.
- Finds source files by the explicit file prefix tag, `SOURCE_TYPE`, and `MOMENTUM_TAG`.
- Averages each dataset path independently inside each cfg.
- Stacks the cfg averages in memory and writes one stripped ensemble file.
- Writes a summary file that records:
  - how many source files were discovered
  - how many were used
  - how many failed
  - how many dataset-level skips happened

### How to use it

- Edit the parameter block at the top of the script.
- Set `INPUT_ROOT` to the directory that contains the cfg folders.
- Set `OUTPUT_ROOT` to the final output root for a given `ENSEMBLE_TAG`.
- Set `CFG_LIST` to the cfg IDs you want to process.
- Set `ENSEMBLE_TAG` and `FILE_TAG` to match the dataset naming scheme.
- Adjust `SOURCE_TYPES` and `MOMENTUM_TAG` for the combinations you want.
- Run the script directly with Python.

### Main knobs

- `SOURCE_TYPES`
- `ENSEMBLE_TAG`
- `FILE_TAG`
- `MOMENTUM_TAG`
- `CFG_LIST`
- `INPUT_ROOT`
- `OUTPUT_ROOT`
- `SOURCE_SINK`
- `OVERWRITE`

### Output layout

- Final ensemble file:
  - `OUTPUT_ROOT / source_type / c2pt_merged_... .h5`
- Summary file:
  - `OUTPUT_ROOT / source_type / summary_... .txt`

### Example command

```bash
python3.11 example/preprocess/strip_pion_c2pt.py
```

## `strip_pion_tmdwf.py`

Strip pion TMDWF source averages from qTMDWF per-configuration HDF5 folders.

### What it does

- Takes an explicit `CFG_LIST`.
- Builds per-cfg input directories from `INPUT_ROOT / cfg`.
- Finds source files with `SOURCE_FILE_GLOB`.
- Averages the full nested HDF5 tree inside each cfg.
- Does not apply the nucleon-style source-time sign flip.
- Writes per-cfg averaged files to `CFG_OUTPUT_ROOT`.
- Combines the per-cfg files into one final ensemble file.
- Writes a summary file that records:
  - how many source files were discovered
  - how many were used
  - how many failed

### How to use it

- Edit the parameter block at the top of the script.
- Set `INPUT_ROOT` to the directory that contains the cfg folders.
- Set `CFG_OUTPUT_ROOT` to the directory for the per-cfg averaged files.
- Set `FINAL_OUTPUT_PATH` to the final HDF5 file location.
- Set `CFG_LIST` to the cfg IDs you want to process.
- Adjust `SOURCE_SINK` and `SOURCE_FILE_GLOB` for the files you want to include.
- Run the script directly with Python.

### Main knobs

- `ENSEMBLE_TAG`
- `MOMENTUM_TAG`
- `FILE_TAG`
- `SOURCE_TYPE`
- `OPERATOR_TAG`
- `CFG_LIST`
- `INPUT_ROOT`
- `CFG_OUTPUT_ROOT`
- `FINAL_OUTPUT_PATH`
- `SUMMARY_PATH`
- `SOURCE_SINK`
- `SOURCE_FILE_GLOB`
- `OVERWRITE`

### Output layout

- Intermediate cfg files:
  - `CFG_OUTPUT_ROOT / <cfg>.h5`
- Final ensemble file:
  - `FINAL_OUTPUT_PATH`
- Summary file:
  - `SUMMARY_PATH`

### Example command

```bash
python3.11 example/preprocess/strip_pion_tmdwf.py
```

## `strip_pion_softff.py`

Strip pion soft form factor source averages from qTMD softFF HDF5 folders.

### What it does

- Searches recursively under `INPUT_ROOT` for source files matching `SOURCE_FILE_GLOB`.
- Averages the entire HDF5 file tree path-by-path across all valid source files.
- Writes one stripped ensemble file directly, without cfg intermediates.
- Writes a summary file that records:
  - how many source files were discovered
  - how many were used
  - how many failed
  - how many dataset-level skips happened

### How to use it

- Edit the parameter block at the top of the script.
- Set `INPUT_ROOT` to the directory that contains the softFF source files.
- Set `OUTPUT_PATH` to the final HDF5 file location.
- Set `SUMMARY_PATH` if you want a separate summary text file.
- Set `ENSEMBLE_TAG` and `MOMENTUM_TAG` to match the dataset naming scheme.
- Run the script directly with Python.

### Main knobs

- `ENSEMBLE_TAG`
- `MOMENTUM_TAG`
- `INPUT_ROOT`
- `OUTPUT_PATH`
- `SUMMARY_PATH`
- `SOURCE_FILE_GLOB`
- `OVERWRITE`

### Example command

```bash
python3.11 example/preprocess/strip_pion_softff.py
```

## `strip_nucleon_tmdpdf.py`

Two-step nucleon TMDPDF 3pt preprocessing with an explicit legacy-style parameter block.

### What it does

- Takes an explicit `CFG_LIST`.
- Loops over `TMDtype`, `projlist / projSave`, `tslist`, `QUARK_FLAVORS`, and `gmlist`.
- Searches each cfg directory for source files matching the current combination.
- Averages source files inside each cfg one by one, using streaming accumulation to keep memory usage low.
- Can restrict source averaging to a selected subset of dataset paths through the explicit momentum-transfer and separation lists.
- When those path patterns are exact, the helper reads only those datasets instead of flattening the full HDF5 tree.
- Applies the source-time sign flip from the antiperiodic temporal boundary
  condition when `APPLY_SOURCE_TIME_SIGN` is enabled.
- Writes per-cfg intermediate HDF5 files.
- Combines the per-cfg files into one final ensemble file.
- Writes a summary file that records:
  - the file type and cfg being processed
  - how many source files were discovered
  - how many were used
  - how many failed

### Practical note

- The nucleon TMDPDF data volume is large, and even a single source file can be big.
- If you are only testing the workflow, prefer a small subset of path selections, for example a single `gm`, a single `ts`, and minimal `qx / qy / qz / bT / bz` ranges.
- If you want to process all data and all paths, it is better to split the work on the cluster and parallelize over `ts`, `cfg`, `gm`, and related combinations at the shell level.

### How to use it

- Edit the parameter block at the top of the script.
- Set `INPUT_ROOT` to the root directory that contains the cfg folders.
- Set `OUTPUT_ROOT` to the root output directory for the stripped files.
- Set `CFG_LIST` to the cfg IDs you want to process.
- Set `ENSEMBLE_TAG` to match the ensemble name used in the input path.
- Adjust `TMDtype`, `projlist`, `projSave`, `tslist`, `QUARK_FLAVORS`, `gmlist`, `qxlist / qylist / qzlist`, `bTdir`, `etalist`, `bTlist`, and `bzlist` for the combinations you want.
- Run the script directly with Python.

### Main knobs

- `MOMENTUM_TAG`
- `ENSEMBLE_TAG`
- `Nt`
- `tslist`
- `projlist`
- `projSave`
- `TMDtype`
- `CFG_LIST`
- `QUARK_FLAVORS`
- `SOURCE_TAG`
- `sm`
- `gmlist`
- `qxlist`
- `qylist`
- `qzlist`
- `bTdir`
- `etalist`
- `bTlist`
- `bzlist`
- `INPUT_ROOT`
- `OUTPUT_ROOT`
- `SOURCE_SINK`
- `APPLY_SOURCE_TIME_SIGN`
- `OVERWRITE`

### Output layout

- Intermediate cfg files:
  - `OUTPUT_ROOT / quark_flavor / tmd_type / proj_save / dt<tsep> / gm / cfg / <cfg>.h5`
- Final ensemble file:
  - `OUTPUT_ROOT / quark_flavor / tmd_type / proj_save / dt<tsep> / gm / qTMD_src... .h5`
- Summary file:
  - `OUTPUT_ROOT / quark_flavor / tmd_type / proj_save / dt<tsep> / gm / summary_... .txt`

### Example command

```bash
python3.11 example/preprocess/strip_nucleon_tmdpdf.py
```
