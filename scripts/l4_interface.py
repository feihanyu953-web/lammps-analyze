#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""l4: interface-specific analysis for aqueous interfaces -- dipole
orientation, adsorbate 2D density (SDF), and water-layer properties.

Modes (--mode orientation | sdf | layer-props | all):
  orientation -- water dipole angle vs the surface normal (z): cos(theta)
                 histogram over all molecules/frames. cos ~ +1 / -1 means
                 dipoles pointing along/against the normal; 0 means lying flat.
  sdf        -- 2D number-density map in the xy plane for an adsorbate
                 element (--adsorbate-element, e.g. K, Na, N of NH3). PBC
                 wrapping in x/y; normalized per-bin area.
  layer-props-- reuses the significant-minima layer split (same algorithm as
                 l2_density) and reports per-layer O counts, thickness and
                 mean density.

Requires unwrapped coordinates (l1 product) and water molecules (O/H types).

Outputs: <name>_orientation.csv/.png, <name>_sdf.csv/.png,
         <name>_layers_props.csv, CONCLUSION line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_io import parse_dump, read_types_elements  # noqa: E402
from l2_density import significant_minima  # noqa: E402
from molecules import find_water_molecules, water_dipoles, water_o_positions  # noqa: E402
from plot_utils import apply_style, print_conclusion, savefig, smooth  # noqa: E402
from region import add_region_args  # noqa: E402


def orientation_cos(mu: np.ndarray) -> np.ndarray:
    """cos(theta) of dipole vs +z for every molecule/frame (flattened)."""
    nrm = np.linalg.norm(mu, axis=2)
    muhat = mu / np.where(nrm > 0, nrm, 1.0)[:, :, None]
    return muhat[:, :, 2].ravel()


def sdf_xy(coords_ads: np.ndarray, box_xy, grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D histogram of adsorbate positions in xy with PBC wrapping.

    coords_ads: (nframe, n_atom, 3); box_xy: (2, 2) lo/hi.
    Returns (H, x_centers, y_centers) with H normalized per bin area (A^-2).
    """
    Lx = box_xy[0, 1] - box_xy[0, 0]
    Ly = box_xy[1, 1] - box_xy[1, 0]
    x = coords_ads[:, :, 0].ravel() % Lx
    y = coords_ads[:, :, 1].ravel() % Ly
    H, xedges, yedges = np.histogram2d(x, y, bins=grid,
                                       range=[[0, Lx], [0, Ly]])
    n_frames = coords_ads.shape[0]
    area_bin = (Lx / grid) * (Ly / grid)
    H = H / (n_frames * area_bin)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    return H, xc, yc


def layer_props(o_pos: np.ndarray, box: np.ndarray, nbins: int = 100,
                smooth_win: int = 5) -> list[dict]:
    """Split O density along z by significant minima; per-layer stats."""
    L = box[2, 1] - box[2, 0]
    z = o_pos[:, :, 2].ravel() % L
    hist, edges = np.histogram(z, bins=nbins, range=(0.0, L))
    centers = 0.5 * (edges[:-1] + edges[1:])
    perp_area = (box[0, 1] - box[0, 0]) * (box[1, 1] - box[1, 0])
    rho = hist / (o_pos.shape[0] * perp_area * (edges[1] - edges[0]))  # A^-3
    mins = significant_minima(smooth(rho, smooth_win))
    bounds = [0.0] + [float(centers[i]) for i in mins] + [float(L)]
    layers = []
    for li in range(len(bounds) - 1):
        lo, hi = bounds[li], bounds[li + 1]
        m = (centers >= lo) & (centers <= hi)
        layers.append({
            "layer": li + 1, "zmin": lo, "zmax": hi,
            "thickness": hi - lo,
            "n_O": int(hist[m].sum()),
            "mean_rho": float(rho[m].mean()) if m.any() else 0.0,
        })
    # drop spurious slivers (noise valleys with ~no atoms): merge into neighbors
    total = max(sum(l["n_O"] for l in layers), 1)
    layers = [l for l in layers if l["n_O"] >= 0.02 * total]
    return layers


def main() -> int:
    ap = argparse.ArgumentParser(description="l4: interface analysis (orientation / sdf / layers)")
    ap.add_argument("--dump", required=True, help="<name>_uw.dump (unwrapped)")
    ap.add_argument("--data", default=None, help="data.lammps (type->element)")
    ap.add_argument("--start", type=int, default=0, help="first frame (default 0)")
    ap.add_argument("--stop", type=int, default=-1, help="last frame, exclusive (default all)")
    ap.add_argument("--stride", type=int, default=1, help="frame stride (default 1)")
    ap.add_argument("--mode", choices=["orientation", "sdf", "layer-props", "all"],
                    default="all")
    ap.add_argument("--adsorbate-element", default=None,
                    help="element for the 2D density map (sdf mode); optional, "
                         "sdf is skipped (with a note) when absent from the map")
    ap.add_argument("--grid", type=int, default=100, help="SDF grid points per axis")
    ap.add_argument("--nbins", type=int, default=90, help="orientation cos(theta) bins")
    ap.add_argument("--elements-map", default=None, help="type->element override")
    ap.add_argument("--outdir", default=".", help="output directory")
    ap.add_argument("--name", default="interface", help="output file prefix")
    add_region_args(ap)
    args = ap.parse_args()

    if args.per_frame:
        raise SystemExit("ERROR: l4 modes use fixed atom identity; use static regions")

    d = parse_dump(args.dump, start=args.start, stop=args.stop, stride=args.stride)
    if d.wrapped:
        raise SystemExit("ERROR: l4 needs unwrapped coordinates; run l1_traj_prep first")
    elements = read_types_elements(data_path=args.data, elements_map=args.elements_map)

    o_types = [t for t, e in elements.items() if e == "O"]
    h_types = [t for t, e in elements.items() if e == "H"]
    if (not o_types or not h_types) and args.mode in ("orientation", "layer-props", "all"):
        raise SystemExit(f"ERROR: need O and H types for orientation/layer-props "
                         f"(got {elements}); pass --data or --elements-map")
    ads_types = [t for t, e in elements.items() if e == args.adsorbate_element]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    conclusions = []

    # ---- orientation ----
    if args.mode in ("orientation", "all"):
        paired = find_water_molecules(d.types, d.coords, o_types[0], h_types,
                                      box=d.box, return_frames=True)
        if paired is None:
            raise SystemExit("ERROR: could not pair water molecules (check O/H types)")
        mols, frames = paired
        mu = water_dipoles(mols, d.coords, box=d.box, frames=frames)
        cos = orientation_cos(mu)
        csv = outdir / f"{args.name}_orientation.csv"
        hist, edges = np.histogram(cos, bins=args.nbins, range=(-1, 1))
        centers = 0.5 * (edges[:-1] + edges[1:])
        with open(csv, "w", encoding="utf-8") as fh:
            fh.write("cos_theta,count\n")
            for i in range(len(centers)):
                fh.write(f"{centers[i]:.5f},{hist[i]}\n")
        print(f"==> wrote {csv}")
        apply_style()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.hist(cos, bins=args.nbins, range=(-1, 1), alpha=0.7)
        ax.axvline(cos.mean(), color="tab:red", lw=0.8, ls="--",
                   label=f"<cos>={cos.mean():.3f}")
        ax.set_xlabel("cos(theta) vs surface normal")
        ax.set_ylabel("count")
        ax.legend(fontsize=9)
        savefig(fig, outdir / f"{args.name}_orientation.png")
        conclusions.append(f"dipole <cos(theta)>={cos.mean():.3f}±{cos.std():.3f} "
                           f"(n={cos.size})")

    # ---- sdf ----
    if args.mode in ("sdf", "all"):
        if not args.adsorbate_element or not ads_types:
            if args.mode == "sdf":
                raise SystemExit(
                    f"ERROR: adsorbate element '{args.adsorbate_element}' not in "
                    f"type map {elements}; pass --adsorbate-element matching "
                    f"--elements-map")
            print(f"==> sdf skipped: no adsorbate element "
                  f"'{args.adsorbate_element}' in map {elements}")
            ads_mask = np.zeros(len(d.atom_ids[0]), dtype=bool)
        else:
            ads_mask = np.isin(d.types[0], ads_types)
        if not ads_mask.any():
            if args.mode == "sdf":
                raise SystemExit(f"ERROR: no atoms of element '{args.adsorbate_element}'")
            print(f"==> sdf skipped: no atoms of element "
                  f"'{args.adsorbate_element}' in the trajectory")
            ads_mask = None
        if ads_mask is not None:
            H, xc, yc = sdf_xy(d.coords[:, ads_mask], d.box[0, :2], args.grid)
            csv = outdir / f"{args.name}_sdf.csv"
            with open(csv, "w", encoding="utf-8") as fh:
                fh.write("x,y,rho\n")
                for i in range(len(xc)):
                    for j in range(len(yc)):
                        fh.write(f"{xc[i]:.5f},{yc[j]:.5f},{H[i, j]:.6g}\n")
            print(f"==> wrote {csv}")
            apply_style()
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5.5, 5))
            im = ax.imshow(H.T, origin="lower",
                           extent=[xc[0], xc[-1], yc[0], yc[-1]],
                           aspect="equal", cmap="viridis")
            fig.colorbar(im, ax=ax,
                         label=f"{args.adsorbate_element} density (A^-2)")
            ax.set_xlabel("x (A)")
            ax.set_ylabel("y (A)")
            savefig(fig, outdir / f"{args.name}_sdf.png")
            conclusions.append(
                f"{args.adsorbate_element} 2D density: max={H.max():.4g} "
                f"A-2, mean={H.mean():.4g} A-2 (grid {args.grid}x{args.grid})")

    # ---- layer-props ----
    if args.mode in ("layer-props", "all"):
        paired = find_water_molecules(d.types, d.coords, o_types[0], h_types,
                                      box=d.box, return_frames=True)
        if paired is None:
            raise SystemExit("ERROR: could not pair water molecules (check O/H types)")
        mols, frames = paired
        o_pos = water_o_positions(mols, d.coords, frames=frames)
        layers = layer_props(o_pos, d.box[frames[0]])
        csv = outdir / f"{args.name}_layers_props.csv"
        with open(csv, "w", encoding="utf-8") as fh:
            fh.write("layer,zmin,zmax,thickness,n_O,mean_rho\n")
            for L in layers:
                fh.write(f"{L['layer']},{L['zmin']:.4f},{L['zmax']:.4f},"
                         f"{L['thickness']:.4f},{L['n_O']},{L['mean_rho']:.6g}\n")
        print(f"==> wrote {csv}")
        conclusions.append(
            f"{len(layers)} water layers: " + "; ".join(
                f"L{l['layer']} z∈[{l['zmin']:.1f},{l['zmax']:.1f}] "
                f"{l['thickness']:.1f}A nO={l['n_O']}" for l in layers))

    print_conclusion("; ".join(conclusions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
