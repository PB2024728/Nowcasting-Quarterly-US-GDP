"""
Shared plotting style and helpers for all project figures.

Call set_style() once at the top of any figure-generating module.
Use save_fig(fig, name) to write PNG (300 DPI) + PDF to results/figures/.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl

from src.config import FIGURES_DIR

# Consistent color palette
COLORS = {
    "AR(p)":              "#6B6B6B",   # neutral gray
    "RW":                 "#AAAAAA",   # light gray
    "bridge_combination": "#2166AC",   # deep blue
    "midas_combination":  "#D6604D",   # orange-red
    "lasso":              "#4DAC26",   # green
    "elasticnet":         "#B2182B",   # crimson
    "dfm":                "#762A83",   # purple
}

# Vintage bar colors (light → dark for months 1 → 3)
VINTAGE_COLORS = ["#9ECAE1", "#4292C6", "#084594"]

MODEL_LABELS = {
    "AR(p)":              "AR(p)",
    "RW":                 "RW",
    "bridge_combination": "Bridge",
    "midas_combination":  "MIDAS",
    "lasso":              "Lasso",
    "lasso_covid":        "Lasso+D",
    "elasticnet":         "ElasticNet",
    "elasticnet_covid":   "ElasticNet+D",
    "dfm":                "DFM",
}


def set_style() -> None:
    """Apply consistent project-wide matplotlib style."""
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "lines.linewidth": 1.5,
        }
    )


def save_fig(fig: plt.Figure, name: str) -> None:
    """Save figure as both PNG (300 DPI) and PDF to results/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png / .pdf")
