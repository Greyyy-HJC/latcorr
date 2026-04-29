"""Placeholder analysis driver for proton quasi PDF.

This module is reserved for a high-level workflow where ``main()`` will:
1. parse paths for multiple correlator inputs;
2. parse per-correlator settings (for example ``gamma`` and ``momentum``);
3. run the analysis pipeline to assemble the proton qPDF observable;
4. produce plots and final bare matrix-element outputs.
"""

from __future__ import annotations


def main() -> None:
    """Placeholder CLI/entrypoint for proton qPDF analysis."""
    raise NotImplementedError(
        "proton_qpdf main() is a placeholder. "
        "Future implementation will load correlators, configure channels, "
        "and output figures plus bare matrix elements."
    )

