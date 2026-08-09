#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""l1: log.lammps thermo parsing -- equilibrium check + block averaging.

Handles variable-column thermo blocks (a 'Step ...' header starts a new block;
rows matching the active header are collected; blocks are merged on the
intersection of their column names). Reports:
  - drift of the energy column (linear slope x window < drift_tol x residual sigma)
  - block averages (--nblocks) with mean +- std
  - multi-panel PNG (T / P / energy) with block means overlaid

Outputs: <name>_thermo.csv, <name>_equil.csv, <name>_thermo.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_utils import apply_style, print_conclusion, savefig  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ENERGY_PRIORITY = ["TotEng", "PotEng", "Enthalpy", "KinEng"]


def parse_thermo(path) -> tuple[dict, list[str]]:
    """Parse log.lammps thermo blocks. Returns (columns_arrays, common_cols).

    columns_arrays: {colname: np.array}; common_cols: columns present in every
    block (merged on intersection, reordered by Step).
    """
    blocks: list[dict[str, np.ndarray]] = []
    header: list[str] | None = None
    rows: list[list[float]] = []

    def flush():
        nonlocal header, rows
        if header is None:
            return
        arr = np.array(rows)
        blocks.append({c: arr[:, i] for i, c in enumerate(header)})
        header, rows = None, []

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            s = raw.strip()
            if not s:
                continue
            toks = s.split()
            if toks and toks[0] == "Step":
                flush()
                header = toks
                rows = []
                continue
            if header is not None and len(toks) == len(header):
                try:
                    rows.append([float(t) for t in toks])
                except ValueError:
                    pass  # stray text line, skip
    flush()

    if not blocks:
        raise ValueError(f"no thermo block (header starting with 'Step') found in {path}")

    common = set(blocks[0])
    for b in blocks[1:]:
        common &= set(b)
    # preserve the first block's column order
    common = [c for c in blocks[0] if c in common]
    if "Step" not in common:
        raise ValueError("thermo blocks share no 'Step' column")

    merged = {}
    for c in common:
        merged[c] = np.concatenate([b[c] for b in blocks])
    order = np.argsort(merged["Step"], kind="stable")
    return {c: merged[c][order] for c in common}, [c for c in common if c != "Step"]


def energy_column(avail: list[str]) -> str | None:
    for cand in ENERGY_PRIORITY:
        if cand in avail:
            return cand
    for c in avail:
        if "Eng" in c or "energy" in c.lower():
            return c
    return None


def drift_check(t, y, tol: float) -> tuple[bool, float, float]:
    """Fit y(t) linearly; return (equilibrated, slope, residual_sigma).

    equilibrated iff |slope| * window_width < tol * residual_sigma.
    """
    slope, intercept = np.polyfit(t, y, 1)
    resid = y - (slope * t + intercept)
    sigma = resid.std()
    window = float(t[-1] - t[0]) if len(t) > 1 else 1.0
    return (abs(slope) * window < tol * sigma, slope, sigma)


def block_average(y, nblocks: int) -> list[tuple[float, float, int, int]]:
    n = len(y)
    edges = np.linspace(0, n, nblocks + 1).astype(int)
    out = []
    for i in range(nblocks):
        seg = y[edges[i]: edges[i + 1]]
        if len(seg):
            out.append((seg.mean(), seg.std(), edges[i], edges[i + 1]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="l1: log.lammps thermo parse + equilibrium")
    ap.add_argument("--log", required=True, help="log.lammps input")
    ap.add_argument("--keys", default=None,
                    help="comma-separated columns to report, e.g. 'Step,Temp,PotEng,Press' "
                         "(default: all common columns)")
    ap.add_argument("--nblocks", type=int, default=5, help="blocks for block averaging")
    ap.add_argument("--drift-tol", type=float, default=0.5,
                    help="drift criterion: |slope|*window < tol*sigma. tol>=0.5 "
                         "passes pure noise (finite-sample slope scatter ~ "
                         "sigma*sqrt(12/N)); tighten for stricter checks")
    ap.add_argument("--outdir", default=".", help="output directory")
    ap.add_argument("--name", default="log", help="output file prefix")
    args = ap.parse_args()

    data, avail = parse_thermo(args.log)
    ecol = energy_column(avail)

    keys = [k.strip() for k in args.keys.split(",")] if args.keys else avail
    missing = [k for k in keys if k not in data]
    if missing:
        print(f"WARNING: requested keys not in thermo columns: {missing}")
        keys = [k for k in keys if k in data]
    if "Step" in keys:
        keys.remove("Step")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = outdir / args.name

    # --- thermo csv ---
    csv_path = f"{prefix}_thermo.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("Step," + ",".join(keys) + "\n")
        for i in range(len(data["Step"])):
            fh.write(",".join(f"{data[k][i]:.8g}" for k in ["Step"] + keys) + "\n")
    print(f"==> wrote {csv_path}")

    # --- equilibrium ---
    t = data["Step"].astype(float)
    equil_lines = []
    overall_ok = None
    if ecol is not None and len(data[ecol]) > 10:
        ok, slope, sigma = drift_check(t, data[ecol], args.drift_tol)
        overall_ok = ok
        equil_lines.append(
            f"{ecol} drift: slope={slope:.3e} {ecol}/step, resid_sigma={sigma:.4f}, "
            f"|slope|*window={abs(slope)*(t[-1]-t[0]):.4f} "
            f"< {args.drift_tol}*sigma={args.drift_tol*sigma:.4f} -> "
            f"{'EQUILIBRATED' if ok else 'NOT equilibrated'}")
    else:
        equil_lines.append(f"no energy column found (avail: {avail}); drift check skipped")

    blocks = {}
    for k in ["Step"] + keys:
        blocks[k] = block_average(data[k], args.nblocks)
    eq_path = f"{prefix}_equil.csv"
    with open(eq_path, "w", encoding="utf-8") as fh:
        fh.write("block,step_start,step_end," + ",".join(f"{k}_mean,{k}_std" for k in keys) + "\n")
        for i in range(args.nblocks):
            row = [str(i), blocks["Step"][i][2], blocks["Step"][i][3]]
            for k in keys:
                mean, std, _, _ = blocks[k][i]
                row.append(f"{mean:.8g}")
                row.append(f"{std:.8g}")
            fh.write(",".join(row) + "\n")
    print(f"==> wrote {eq_path}")
    for line in equil_lines:
        print("    " + line)

    # --- png ---
    apply_style()
    npanels = sum(1 for k in ("Temp", "Press") if k in data) + (1 if ecol else 0)
    if npanels:
        fig, axes = plt.subplots(npanels, 1, figsize=(8, 2.6 * npanels), sharex=True)
        axes = np.atleast_1d(axes)
        ip = 0
        t_ps = None
        for key, ylab in (("Temp", "Temperature (K)"), ("Press", "Pressure (bar)")):
            if key in data:
                ax = axes[ip]; ip += 1
                ax.plot(t, data[key], lw=0.6)
                for b in blocks[key]:
                    ax.axhline(b[0], color="tab:red", lw=0.8, ls="--", alpha=0.7)
                ax.set_ylabel(ylab)
        if ecol:
            ax = axes[ip]
            ax.plot(t, data[ecol], lw=0.6)
            for b in blocks[ecol]:
                ax.axhline(b[0], color="tab:red", lw=0.8, ls="--", alpha=0.7)
            ax.set_ylabel(f"{ecol} (energy units)")
        axes[-1].set_xlabel("Step")
        png = f"{prefix}_thermo.png"
        savefig(fig, png)

    if overall_ok is not None:
        summary = f"block-averaged " + "; ".join(
            f"{k}={np.mean([b[0] for b in blocks[k]]):.4g}±{np.std([b[0] for b in blocks[k]]):.4g}"
            for k in keys[:3])
        verdict = "equilibrated" if overall_ok else "NOT equilibrated"
        print_conclusion(f"{Path(args.log).name}: {verdict} ({ecol} drift criterion); {summary}")
    else:
        print_conclusion(f"{Path(args.log).name}: equilibrium check skipped (no energy column)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
