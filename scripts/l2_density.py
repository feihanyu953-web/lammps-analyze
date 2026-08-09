#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""l2: 1D density profile (number / mass) along an axis + water-layer detection.

Pure numpy on the parsing layer (dump_io): coordinates are histogrammed per
frame with PBC wrapping (mod box length) and normalized by the per-frame bin
volume, so NPT boxes with fluctuating cell sizes are handled. MDAnalysis's
DensityAnalysis is not used: it produces a 3D grid, while our deliverable is
a 1D profile (see references/method_notes.md).

Water layers are found from the smoothed rho(z): local minima split the
profile; layers.csv records (layer, zmin, zmax, thickness, particle count,
mean density).

Outputs: <name>_density.csv, <name>_layers.csv, <name>_density.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_io import parse_dump, read_types_elements  # noqa: E402
from plot_utils import apply_style, print_conclusion, savefig, smooth  # noqa: E402
from region import add_region_args, select_traj  # noqa: E402

MASSES = {"H": 1.008, "O": 15.999, "N": 14.007, "C": 12.011, "S": 32.06,
          "Na": 22.990, "K": 39.098, "Li": 6.941, "Cs": 132.905, "Cl": 35.45}


def local_minima(y: np.ndarray):
    """Indices of strict local minima in a 1D array."""
    out = []
    for i in range(1, len(y) - 1):
        if y[i] < y[i - 1] and y[i] <= y[i + 1]:
            out.append(i)
    return np.array(out, dtype=int)


def significant_minima(y: np.ndarray, frac: float = 0.3):
    """Local minima whose depth below the lower neighboring peak exceeds
    frac * peak -- filters out noise ripples (e.g. Gaussian tails)."""
    mins = local_minima(y)
    keep = []
    for i in mins:
        left_max = y[:i].max() if i > 0 else y[i]
        right_max = y[i + 1:].max() if i + 1 < len(y) else y[i]
        peak = min(left_max, right_max)
        if peak > 0 and (peak - y[i]) > frac * peak:
            keep.append(i)
    return np.array(keep, dtype=int)


def main() -> int:
    ap = argparse.ArgumentParser(description="l2: 1D density profile + water layers")
    ap.add_argument("--dump", required=True, help="dump.lammpstrj or <name>_uw.dump")
    ap.add_argument("--data", default=None, help="data.lammps (type->element)")
    ap.add_argument("--start", type=int, default=0, help="first frame (default 0)")
    ap.add_argument("--stop", type=int, default=-1, help="last frame, exclusive (default all)")
    ap.add_argument("--stride", type=int, default=1, help="frame stride (default 1)")
    ap.add_argument("--axis", default="z", choices=["x", "y", "z"],
                    help="profile axis (default z)")
    ap.add_argument("--nbins", type=int, default=200, help="number of bins")
    ap.add_argument("--density-type", choices=["number", "mass"], default="number",
                    help="number density (A^-3) or mass density (g/cm^3)")
    ap.add_argument("--types", default=None, help="type ids to include, e.g. '1 2' "
                                                  "(default: all)")
    ap.add_argument("--smooth", type=int, default=5,
                    help="moving-average window for layer detection (odd; 1 = off)")
    ap.add_argument("--layers", action="store_true",
                    help="detect layers by local minima of rho(z)")
    ap.add_argument("--elements-map", default=None, help="type->element override")
    ap.add_argument("--outdir", default=".", help="output directory")
    ap.add_argument("--name", default="density", help="output file prefix")
    add_region_args(ap)
    args = ap.parse_args()

    d = parse_dump(args.dump, start=args.start, stop=args.stop, stride=args.stride)
    axis = {"x": 0, "y": 1, "z": 2}[args.axis]
    elements = read_types_elements(data_path=args.data, elements_map=args.elements_map)
    type_filter = None
    if args.types:
        type_filter = {int(t) for t in str(args.types).split()}
        keep = np.isin(d.types[0], list(type_filter))
    else:
        keep = np.ones(d.natoms, dtype=bool)

    mask = select_traj(d.coords, d.box, args, d.types)
    # effective per-atom selection: type filter AND region (static or per-frame)
    if args.per_frame:
        sel = mask & np.broadcast_to(keep, mask.shape)
    else:
        sel = np.broadcast_to(keep & mask[0], (len(d.timesteps), d.natoms))

    # histogram with PBC wrapping (mod box) and per-frame volume normalization
    edges = None
    counts = None
    mass_bins = None
    vol_sum = 0.0
    for f in range(len(d.timesteps)):
        L = d.box[f, axis, 1] - d.box[f, axis, 0]
        m = sel[f]
        if not m.any():
            continue
        z = d.coords[f, m, axis] % L  # PBC wrap
        if edges is None:
            edges = np.linspace(0.0, L, args.nbins + 1)
        hist, _ = np.histogram(z, bins=edges)
        counts = hist if counts is None else counts + hist
        perp_area = np.prod([d.box[f, j, 1] - d.box[f, j, 0] for j in range(3) if j != axis])
        vol_sum += perp_area * (edges[1] - edges[0])
        if args.density_type == "mass":
            masses = np.array([MASSES.get(elements.get(int(t), ""), 0.0)
                               for t in d.types[f, m]])
            hist_m, _ = np.histogram(z, bins=edges, weights=masses)  # mass per bin
            mass_bins = hist_m if mass_bins is None else mass_bins + hist_m

    centers = 0.5 * (edges[:-1] + edges[1:])
    if counts is None or counts.sum() == 0:
        raise SystemExit("ERROR: no atoms selected for density profile")

    if args.density_type == "number":
        rho = counts / vol_sum  # A^-3
        ylabel = f"number density (A$^{{-3}}$)"
    else:
        if mass_bins is None or float(mass_bins.sum()) == 0:
            raise SystemExit(
                "ERROR: --density-type mass needs a type->element map (mass 0 for "
                "all atoms); pass --data data.lammps or --elements-map '1:O 2:H ...'")
        # mass per bin / bin volume; 1 u/A^3 = 1.66054 g/cm^3
        rho = mass_bins / vol_sum * 1.66054
        ylabel = "mass density (g/cm$^3$)"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{args.name}_density.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(f"{args.axis}(A),rho\n")
        for i in range(len(centers)):
            fh.write(f"{centers[i]:.5f},{rho[i]:.8g}\n")
    print(f"==> wrote {csv_path}")

    # --- layers ---
    layers_txt = []
    if args.layers:
        ys = smooth(rho, args.smooth)
        minima = significant_minima(ys)
        # split into layers at minima; require a minimum depth relative to neighbors
        bounds = [0.0] + [float(centers[i]) for i in minima] + [float(centers[-1])]
        layers_csv = outdir / f"{args.name}_layers.csv"
        with open(layers_csv, "w", encoding="utf-8") as fh:
            fh.write("layer,zmin(A),zmax(A),thickness(A),particle_count,mean_rho\n")
            for li in range(len(bounds) - 1):
                lo, hi = bounds[li], bounds[li + 1]
                in_layer = (centers >= lo) & (centers <= hi)
                n_part = int(counts[in_layer].sum())
                mean_rho = float(rho[in_layer].mean()) if in_layer.any() else 0.0
                fh.write(f"{li + 1},{lo:.4f},{hi:.4f},{hi - lo:.4f},{n_part},{mean_rho:.6g}\n")
                layers_txt.append(f"L{li + 1}: z in [{lo:.2f},{hi:.2f}] A, "
                                  f"thick {hi - lo:.2f} A, {n_part} atoms")
        print(f"==> wrote {layers_csv}")

    # --- png ---
    apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(centers, rho, lw=1.0)
    if args.layers:
        ys = smooth(rho, args.smooth)
        minima = significant_minima(ys)
        for i in minima:
            ax.axvline(centers[i], color="tab:red", lw=0.7, ls="--")
        ax.set_title(f"layers: {len(minima) + 1} detected", fontsize=9)
    ax.set_xlabel(f"{args.axis} (A)")
    ax.set_ylabel(ylabel)
    png = outdir / f"{args.name}_density.png"
    savefig(fig, png)

    if layers_txt:
        print_conclusion(f"{len(layers_txt)} layers detected: " + "; ".join(layers_txt))
    else:
        print_conclusion(f"density profile along {args.axis}: "
                         f"mean rho={float(rho.mean()):.4g} "
                         f"({args.density_type} density; add --layers for layer splitting)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
