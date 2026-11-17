"""Streamlit entry point of the PollenFormerUCO demo: ``streamlit run demo/main.py``.

The UI mode comes from ``--mode`` (``streamlit run demo/main.py -- --mode HARD``), then the
``UI_MODE`` environment variable, then defaults to the public-facing ``SOFT`` mode.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path

import streamlit as st

LOGO = Path(__file__).resolve().parent / "static" / "ava_logo.png"

# Hide Streamlit's deploy button and main menu: this is a public demo, not a dev server.
_HIDE_BUTTONS = """
    <style>
        button[data-testid="stBaseButton-header"] { display: none !important; }
        #MainMenu { display: none !important; }
    </style>
"""


def resolve_mode(argv: Sequence[str] | None = None) -> str:
    """Return the UI mode: ``--mode`` wins, then ``UI_MODE``, then ``SOFT``.

    Args:
        argv: Arguments to parse; defaults to ``sys.argv[1:]``. Unknown ones are ignored.

    Returns:
        ``"SOFT"`` or ``"HARD"``.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", type=str.upper, choices=["HARD", "SOFT"], default=None)
    args, _ = parser.parse_known_args(argv)
    mode = (args.mode or os.getenv("UI_MODE") or "SOFT").upper()
    return mode if mode in {"HARD", "SOFT"} else "SOFT"


def main() -> None:
    """Render the PollenFormerUCO interface for the selected mode."""
    mode = resolve_mode()
    st.set_page_config(
        page_title="PollenFormerUCO Classifier", page_icon=str(LOGO), layout="centered"
    )
    st.markdown(_HIDE_BUTTONS, unsafe_allow_html=True)
    st.sidebar.image(str(LOGO), width="stretch")

    st.title("🔬 PollenFormerUCO · Pollen Image Classifier")
    st.write(
        "Upload a microscopy image and get pollen taxon predictions from PollenFormerUCO "
        "(DINOv2-ViT Base + hybrid CE/Triplet loss, 128-D embedding, k-NN inference)."
    )

    if mode == "HARD":
        from ui.hard import render_ui
    else:
        from ui.soft import render_ui
    render_ui()


if __name__ == "__main__":
    main()
