#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""l2: structural order parameters -- Steinhardt q4/q6, tetrahedral order,
local structure index (LSI), from first-shell neighbors (pure numpy + scipy).

For every selected atom (e.g. water O), the first coordination shell is found
by distance (--cutoff, default 3.5 A, with minimum-image PBC on orthorhombic
cells). Per atom per frame:
  q_l = sqrt(4*pi/(2l+1) * sum_m |mean_j Y_lm(rhat_ij)|^2)     (l = 4, 6)
  q_tet = 1 - 3/8 * sum_{j<k} (cos psi_jk + 1/3)^2             (4 nearest)
  LSI   = rms deviation of the 4 nearest-neighbour distances
Reported as trajectory means + distributions (ice/water discrimination:
tetrahedral ice q6 ~ 0.5-0.6, liquid water ~ 0.2-0.4; LSI water ~ 2.9 A^2).

Outputs: <name>_order.csv (per-frame means), <name>_order.png (histograms),
         <name>_lsi.png (LSI distribution), CONCLUSION line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_io import parse_dump, read_types_elements  # noqa: E402
from plot_utils import apply_style, print_conclusion, savefig  # noqa: E402
from region import add_region_args, select_traj  # noqa: E402


def min_image(dist_vecs, box):
    """Apply minimum-image correction for orthorhombic cells.

    dist_vecs: (..., 3) position differences; box: (3, 2) lo/hi.
    """
    L = box[:, 1] - box[:, 0]
    return dist_vecs - np.round(dist_vecs / L) * L


def neighbor_indices(coords_frame, types_frame, atom_ids_frame, center_ids,
                     center_types, cutoff, box):
    """First-shell neighbor indices (incl. PBC min-image) for center atoms.

    Neighbors are restricted to the same element type(s) as the center atoms
    (order parameters describe the like-atom coordination shell, e.g. O-O).
    Center atoms are addressed by their LAMMPS ids: dumps without 'sort id'
    can reorder rows between frames, so row indices are NOT stable.
    Returns (center_rows, nb_list): the global row of each center atom in this
    frame, and its neighbor rows (arrays of global row indices).
    """
    cand = np.flatnonzero(np.isin(types_frame, center_types))
    d = coords_frame[cand][:, None, :] - coords_frame[cand][None, :, :]
    d = min_image(d, box)
    dist = np.linalg.norm(d, axis=2)
    center_rows = np.empty(len(center_ids), dtype=int)
    nb_list = []
    for j, cid in enumerate(center_ids):
        hit = np.flatnonzero(atom_ids_frame == cid)
        if len(hit) == 0:
            center_rows[j] = -1
            nb_list.append(np.array([], dtype=int))
            continue
        # map the global row to the candidate-subset index (cand is sorted)
        row = int(hit[0])
        center_rows[j] = row
        k = int(np.searchsorted(cand, row))
        nb_rel = np.flatnonzero((dist[k] < cutoff) & (np.arange(len(cand)) != k))
        nb_list.append(cand[nb_rel])
    return center_rows, nb_list


def q_steinhardt(coords_frame, center_idx, nb_list, l: int, box):
    """q_l per center atom for spherical-harmonic rank l (scipy)."""
    # scipy >= 1.15: sph_harm_y(n, m, theta, phi) (old sph_harm(m, l, phi, theta)
    # was removed in scipy 1.17)
    from scipy.special import sph_harm_y

    out = np.zeros(len(center_idx))
    for k, ci in enumerate(center_idx):
        nb = nb_list[k]
        if len(nb) < 2:
            continue
        r = min_image(coords_frame[nb] - coords_frame[ci], box)
        r /= np.linalg.norm(r, axis=1)[:, None]
        theta = np.arccos(np.clip(r[:, 2], -1, 1))
        phi = np.arctan2(r[:, 1], r[:, 0])
        acc = 0.0
        for m in range(-l, l + 1):
            ylm = sph_harm_y(l, m, theta, phi)  # (n_neigh,)
            qlm = ylm.mean()
            acc += abs(qlm) ** 2
        out[k] = np.sqrt(4.0 * np.pi / (2 * l + 1) * acc)
    return out


def q_tetrahedral(coords_frame, center_idx, nb_list, box, n_nearest=4):
    """Tetrahedral order of the n_nearest neighbours: 1 = ideal tetrahedron."""
    out = np.zeros(len(center_idx))
    for k, ci in enumerate(center_idx):
        nb = nb_list[k]
        if len(nb) < n_nearest:
            continue
        r = min_image(coords_frame[nb[:n_nearest]] - coords_frame[ci], box)
        r /= np.linalg.norm(r, axis=1)[:, None]
        c = np.clip(r @ r.T, -1.0, 1.0)
        tri = 0.0
        for j in range(n_nearest):
            for i2 in range(j + 1, n_nearest):
                tri += (c[j, i2] + 1.0 / 3.0) ** 2
        out[k] = 1.0 - 3.0 / 8.0 * tri
    return out


def lsi(coords_frame, center_idx, nb_list, box, n_nearest=4):
    """Local structure index: rms deviation of the n_nearest neighbour distances."""
    out = np.zeros(len(center_idx))
    for k, ci in enumerate(center_idx):
        nb = nb_list[k]
        if len(nb) < n_nearest:
            continue
        r = min_image(coords_frame[nb[:n_nearest]] - coords_frame[ci], box)
        d = np.linalg.norm(r, axis=1)
        out[k] = float(np.sqrt(((d - d.mean()) ** 2).mean()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="l2: order parameters (q4/q6/tet/LSI)")
    ap.add_argument("--dump", required=True, help="dump.lammpstrj or <name>_uw.dump")
    ap.add_argument("--data", default=None, help="data.lammps (type->element)")
    ap.add_argument("--start", type=int, default=0, help="first frame (default 0)")
    ap.add_argument("--stop", type=int, default=-1, help="last frame, exclusive (default all)")
    ap.add_argument("--stride", type=int, default=1, help="frame stride (default 1)")
    ap.add_argument("--elements", default="O",
                    help="element to analyse (default O; needs type map)")
    ap.add_argument("--cutoff", type=float, default=3.5,
                    help="first-shell cutoff, A (default 3.5)")
    ap.add_argument("--order", choices=["q4", "q6", "tet", "all"], default="all")
    ap.add_argument("--lsi", action="store_true", help="compute LSI distribution")
    ap.add_argument("--nbins", type=int, default=60, help="histogram bins")
    ap.add_argument("--elements-map", default=None, help="type->element override")
    ap.add_argument("--outdir", default=".", help="output directory")
    ap.add_argument("--name", default="order", help="output file prefix")
    add_region_args(ap)
    args = ap.parse_args()

    if args.per_frame:
        raise SystemExit("ERROR: order parameters use static atom identity "
                         "(neighbors vary per frame anyway); drop --per-frame")

    d = parse_dump(args.dump, start=args.start, stop=args.stop, stride=args.stride)
    elements = read_types_elements(data_path=args.data, elements_map=args.elements_map)
    tset = [t for t, e in elements.items() if e == args.elements]
    if not tset:
        raise SystemExit(
            f"ERROR: element '{args.elements}' not in type map {elements}; "
            "pass --data or --elements-map")
    center_mask = np.isin(d.types[0], tset)

    mask = select_traj(d.coords, d.box, args, d.types)[0]
    keep = center_mask & mask
    if not keep.any():
        raise SystemExit("ERROR: element+region selection is empty")
    center_ids = d.atom_ids[0, keep]
    print(f"    analysing {len(center_ids)} {args.elements} atoms, "
          f"cutoff={args.cutoff} A, {len(d.timesteps)} frames")

    nframe = len(d.timesteps)
    want_q4 = args.order in ("q4", "all")
    want_q6 = args.order in ("q6", "all")
    want_tet = args.order in ("tet", "all")

    frame_means = {k: [] for k in
                   (["q4"] if want_q4 else []) + (["q6"] if want_q6 else []) +
                   (["tet"] if want_tet else []) + (["lsi"] if args.lsi else [])}
    all_vals = {k: [] for k in frame_means}

    for f in range(nframe):
        center_rows, nb = neighbor_indices(d.coords[f], d.types[f], d.atom_ids[f],
                                           center_ids, tset, args.cutoff, d.box[f])
        if want_q4:
            v = q_steinhardt(d.coords[f], center_rows, nb, 4, d.box[f])
            frame_means["q4"].append(v.mean()); all_vals["q4"].append(v)
        if want_q6:
            v = q_steinhardt(d.coords[f], center_rows, nb, 6, d.box[f])
            frame_means["q6"].append(v.mean()); all_vals["q6"].append(v)
        if want_tet:
            v = q_tetrahedral(d.coords[f], center_rows, nb, d.box[f])
            frame_means["tet"].append(v.mean()); all_vals["tet"].append(v)
        if args.lsi:
            v = lsi(d.coords[f], center_rows, nb, d.box[f])
            frame_means["lsi"].append(v.mean()); all_vals["lsi"].append(v)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    keys = list(frame_means)
    csv_path = outdir / f"{args.name}_order.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("frame," + ",".join(keys) + "\n")
        for f in range(nframe):
            fh.write(f"{f}," + ",".join(f"{frame_means[k][f]:.6g}" for k in keys) + "\n")
    print(f"==> wrote {csv_path}")

    apply_style()
    import matplotlib.pyplot as plt
    nplots = len(keys) + (1 if args.lsi else 0)
    if nplots:
        fig, axes = plt.subplots(nplots, 1, figsize=(6.5, 2.8 * nplots))
        axes = np.atleast_1d(axes)
        ip = 0
        for k in keys:
            ax = axes[ip]; ip += 1
            v = np.concatenate(all_vals[k])
            # constant data (e.g. no neighbours -> all 0) cannot be histogrammed
            bins = args.nbins if v.max() > v.min() else 1
            ax.hist(v, bins=bins, alpha=0.7)
            ax.axvline(np.mean(v), color="tab:red", lw=0.8, ls="--")
            ax.set_xlabel(k)
            ax.set_ylabel("count")
        png = outdir / f"{args.name}_order.png"
        savefig(fig, png)

    means = {k: float(np.mean(np.concatenate(all_vals[k]))) for k in keys}
    stds = {k: float(np.std(np.concatenate(all_vals[k]))) for k in keys}
    parts = [f"{k}={means[k]:.3f}±{stds[k]:.3f}" for k in keys]
    print_conclusion(f"{args.elements} order params: " + "; ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
