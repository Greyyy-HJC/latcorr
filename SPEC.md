# latcorr — project specification

## What this is

**latcorr** is a Python package for **lattice QCD correlator processing and
analysis**. It covers preprocessing pipelines, resampling, correlator readers
and derived observables, ground-state fits, and plotting helpers.

This repository is library code plus lightweight examples/tests. Analysis
scripts should stay thin and use the package APIs directly.

## Layout

- **`latcorr.preprocess`**: preprocessing helpers for cleaning, slicing,
  symmetrizing, source averaging, and ensemble HDF5 construction for nucleon
  and pion workflows.
- **`latcorr.resampling`**: binning, bootstrap, jackknife, and related
  resampling utilities.
- **`latcorr.correlators`**: correlator readers and observable builders,
  including 2pt, 3pt, qDA/TMDWF, 3pt ratios, summed ratios, and FH data.
- **`latcorr.ground_state`**: priors, fit functions, and ground-state fit
  helpers for 2pt, 3pt ratio, FH, ratio+FH joint, qDA/TMDWF, and form-factor
  workflows.
- **`latcorr.plotting`**: plotting helpers for correlator data, ratios,
  effective masses, FH, qDA ratios, and form-factor fits.
- **`latcorr.analysis`**: high-level analysis entrypoints and placeholders
  for workflow-specific scripts.
- **`latcorr.utils`**: small shared utilities such as logging and converters.
- **`example/`**: runnable examples. Prefer top-level workflow steps and small
  local helpers over wrapping every example in a large framework.

## Code style

- Prefer **straight-line, readable** code over deep abstraction trees.
- Avoid thin helpers whose only job is to call another function one level down;
  avoid long chains of delegation.
- Avoid blanket `try`/`except` around normal control flow; use exceptions where
  failures are truly exceptional and unexpected.
- For internal numerics, prefer letting Python/NumPy raise natural errors. Do
  not add explicit `ValueError`/manual validation branches for shape, type, or
  range checks that NumPy/Python will already fail on.
- Keep numerical routines separate from plotting when practical.
- Keep public APIs small and analysis-oriented. Add modules/functions when
  there is real reusable analysis code to put there.
- Prefer direct type hints like `np.ndarray`, `int`, and `bool` when they make
  signatures clearer. Avoid helper type modules, dataclass-heavy configuration,
  protocols, and framework-like structure until the code actually needs them.
- Examples should be easy to run and inspect; keep constants, priors, plotting,
  and workflow steps close to where they are used.

See also the human-oriented overview in [README.md](README.md).
