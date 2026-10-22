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
    """Centered moving average (odd window); same length, edge-safe.

    Windows shrink at the sequence boundaries (each point is averaged over the
    points that actually exist around it). Zero-padded convolution ('same') is
    deliberately NOT used: it divides by the full window even where fewer real
    points exist, faking rising/falling ramps at both edges of stationary time
    series (found on hbond-count plots, lammps2 KCl analysis 2026-10-22).
    """
    y = np.asarray(y, dtype=float)
    if window <= 1 or y.size == 0:
        return y.copy()
    c = np.cumsum(np.insert(y, 0, 0.0))
    idx = np.arange(y.size)
    lo = np.maximum(0, idx - window // 2)
    hi = np.minimum(y.size, idx + window // 2 + 1)
    return (c[hi] - c[lo]) / (hi - lo)
