#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared plotting / reporting helpers for lammps-analyze (matplotlib)."""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless-safe regardless of environment
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def apply_style() -> None:
    plt.rcParams.update(STYLE)


def savefig(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"==> wrote {path}")


def print_conclusion(text: str) -> None:
    """Single machine-readable conclusion line; AI must relay it verbatim."""
    print(f"CONCLUSION: {text}")


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average (odd window); returns a copy of the same length."""
    if window <= 1:
        return np.asarray(y, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(y, dtype=float), kernel, mode="same")
