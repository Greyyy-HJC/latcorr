# latcorr

`latcorr` is a Python package skeleton for lattice QCD data processing and
analysis. The initial layout follows the separation used in LaMETLat:
preprocessing, resampling, correlator construction, plotting, and ground-state
extraction live in separate modules.

## Installation

With `uv`:

```bash
uv sync --dev
```

Then run commands inside the project environment:

```bash
uv run python
uv run pytest
```

Equivalently, use `.venv/bin/python` directly. Running plain `python` from the
outer conda environment does not use the `.venv` created by `uv sync`.

Fallback with `pip`:

```bash
python -m pip install -e ".[dev]"
```

## Package Structure

- `latcorr.preprocess`: preprocessing helpers for cleaning, slicing, symmetrizing, and nucleon 2pt source averaging
- `latcorr.resampling`: binning, bootstrap, and jackknife placeholders
- `latcorr.correlators`: correlator readers (`read_pt2_h5`, `read_qda_h5`)
- `latcorr.analysis`: high-level physics-analysis entrypoints (placeholder scripts)
- `latcorr.plotting`: plotting namespace
- `latcorr.ground_state`: ground-state extraction namespace
- `latcorr.utils`: shared utility namespace

## Minimal Usage

```python
import latcorr
from latcorr.preprocess import preprocess_correlator
from latcorr.resampling import bootstrap, jackknife

print(latcorr.__version__)
```

Example preprocessing workflow:

```python
from latcorr.preprocess import preprocess_correlator
```

See `example/preprocess/` for a synthetic nucleon TMDPDF preprocessing demo.
There is also a nucleon 2pt stripping example that averages multiple sources
per configuration before writing the stripped ensemble HDF5 file.
For the larger 3pt TMDPDF workflow, see `example/preprocess/strip_nucleon_tmdpdf.py`.

## Development

Run tests with:

```bash
python -m pytest
```

## Code Style

- Prefer simple functions and NumPy operations over early abstractions.
- Use direct type hints like `np.ndarray`, `int`, and `bool` when they clarify
  a function signature.
- Avoid helper type modules, dataclass-based configuration, protocols, and
  other framework-like structure until the code actually needs them.
- Keep subpackages lightweight: add files and public APIs only when there is
  real analysis code to put there.
- Keep numerical routines separate from plotting and notebook visualization.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
