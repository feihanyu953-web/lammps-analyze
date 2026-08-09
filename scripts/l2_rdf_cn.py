#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""l2: RDF g(r) + coordination number CN(r) via MDAnalysis InterRDF.

Pairs are given as element names ("O:O", "O:H") resolved through the
type->element map (data.lammps Masses '# element' comments or --elements-map).
Region selection: static first-frame mask by default; --per-frame uses a
direct distance_array histogram (slower; only for RDF/CN).

CN(r) = integral of g(r') 4 pi rho r'^2 up to --cn-cut; the average density
is taken per-frame (NPT-safe). Peak detection ports MD-Analyzer's _find_peaks
(first peak in [1.5, 5.0] A, first valley in [r_peak+0.3, r_peak+1.5]).

Outputs per pair: <pair>_rdf.csv (r, g, CN), rdf_cn.png, CONCLUSION line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_io import read_types_elements  # noqa: E402
from plot_utils import apply_style, print_conclusion, savefig  # noqa: E402
from region import add_region_args, select_frame  # noqa: E402


def load_universe(dump_path, data_path):
    """Topology from data.lammps (DATA) + trajectory as LAMMPSDUMP.

    Passing format="LAMMPS" to Universe() only selects the topology reader;
    the coordinate reader then falls back to a default and misreads the dump.
    """
    import MDAnalysis as mda

    if data_path:
        u = mda.Universe(data_path, format="DATA")
        u.load_new(dump_path, format="LAMMPSDUMP")
    else:
        u = mda.Universe(dump_path, format="LAMMPSDUMP")
    return u


def type_selection(u, types: list[int]) -> str:
    """MDAnalysis selection string for a set of LAMMPS type ids."""
    return "type " + " ".join(str(t) for t in types)


def resolve_element_types(elements: dict[int, str], u, element: str) -> list[int]:
    """Map an element symbol to the LAMMPS type ids carrying it."""
    types = [t for t, el in elements.items() if el == element]
    if not types:
        raise ValueError(
            f"element '{element}' not found in type map {elements}; "
            "use --elements-map (e.g. '1:O 2:H') or provide data.lammps with "
            "'# element' comments in the Masses section")
    return types


def compute_static(u, g1_sel, g2_sel, args) -> tuple[np.ndarray, np.ndarray, float, float]:
    """InterRDF over the trajectory; returns (r, g, n_ref, n_tgt)."""
    from MDAnalysis.analysis.rdf import InterRDF

    rdf = InterRDF(g1_sel, g2_sel, nbins=args.nbins,
                   range=(args.rmin, args.rmax), norm="rdf")
    rdf.run(start=args.start or 0, stop=args.stop, step=args.stride, verbose=False)
    return rdf.results.bins, rdf.results.rdf, len(g1_sel), len(g2_sel)


def main() -> int:
    ap = argparse.ArgumentParser(description="l2: RDF + coordination number (MDAnalysis)")
    ap.add_argument("--dump", required=True, help="dump.lammpstrj or <name>_uw.dump")
    ap.add_argument("--data", default=None, help="data.lammps (type->element map)")
    ap.add_argument("--start", type=int, default=None,
                    help="first frame for InterRDF run (default: all)")
    ap.add_argument("--stop", type=int, default=None,
                    help="last frame, exclusive (default: all)")
    ap.add_argument("--stride", type=int, default=1, help="frame stride (default 1)")
    ap.add_argument("--pairs", default="O:O",
                    help="element pairs separated by spaces, e.g. 'O:O O:H'")
    ap.add_argument("--rmin", type=float, default=0.5, help="r lower bound (A)")
    ap.add_argument("--rmax", type=float, default=10.0, help="r upper bound (A)")
    ap.add_argument("--nbins", type=int, default=300, help="number of r bins")
    ap.add_argument("--cn-cut", type=float, default=3.5,
                    help="integration cutoff for CN(r) (A)")
    ap.add_argument("--elements-map", default=None, help="type->element override")
    ap.add_argument("--outdir", default=".", help="output directory")
    ap.add_argument("--name", default="rdf", help="output file prefix")
    add_region_args(ap)
    args = ap.parse_args()

    if args.per_frame:
        raise SystemExit(
            "ERROR: --per-frame RDF is not implemented in this MVP; drop --per-frame "
            "(static first-frame region selection is the default and is what "
            "InterRDF supports).")

    u = load_universe(args.dump, args.data)
    elements = read_types_elements(data_path=args.data, elements_map=args.elements_map)

    pairs = [p.split(":") for p in args.pairs.replace(":", ":").split()]
    pairs = [(a.strip(), b.strip()) for a, b in pairs]
    if any(len(p) != 2 for p in pairs):
        raise SystemExit("ERROR: --pairs entries must look like 'O:O'")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # static region mask from the first frame (atom indices -> ids)
    u.trajectory[0]
    dims = u.dimensions[:3]
    args_box = np.array([[0, dims[0]], [0, dims[1]], [0, dims[2]]])
    mask = np.ones(len(u.atoms), dtype=bool)
    if args.region != "full" or args.layer_auto:
        mask = select_frame(u.atoms.positions, args_box, args,
                            u.atoms.types.astype(int))
        if not mask.any():
            raise SystemExit("ERROR: region mask selects zero atoms")
        print(f"    region keeps {mask.sum()}/{len(mask)} atoms")
    ids_keep = u.atoms.ids[mask]

    import matplotlib.pyplot as plt
    apply_style()

    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(7, 3.6 * n_pairs))
    if n_pairs == 1:
        axes = [axes]

    conclusions = []
    for ax, (el1, el2) in zip(axes, pairs):
        t1 = resolve_element_types(elements, u, el1)
        t2 = resolve_element_types(elements, u, el2)
        base1 = u.select_atoms(type_selection(u, t1))
        base2 = u.select_atoms(type_selection(u, t2))
        sel1 = base1[np.isin(base1.ids, ids_keep)]
        sel2 = base2[np.isin(base2.ids, ids_keep)]
        if len(sel1) == 0 or len(sel2) == 0:
            raise SystemExit(
                f"ERROR: pair {el1}:{el2} has an empty selection "
                f"(types {t1}/{t2}, {len(sel1)}/{len(sel2)} atoms)")

        r, g, n_ref, n_tgt = compute_static(u, sel1, sel2, args)

        # CN(r) with per-frame density (NPT-safe)
        vol = []
        for ts in u.trajectory:
            vol.append(np.prod(ts.dimensions[:3]))
        rho_tgt = n_tgt / float(np.mean(vol))  # target atoms per A^3
        dr = r[1] - r[0]
        cn = np.zeros_like(g)
        for i in range(len(r)):
            cn[i] = 4.0 * np.pi * rho_tgt * np.trapezoid(g[: i + 1] * r[: i + 1] ** 2,
                                                         r[: i + 1]) if i else 0.0
        cn_at_cut = cn[np.searchsorted(r, args.cn_cut) - 1] if r[0] < args.cn_cut else 0.0

        # first peak / first valley (MD-Analyzer _find_peaks port, plus a
        # significance gate: the peak must exceed 2x the window mean, so a
        # featureless g(r) does not report a spurious peak)
        pk_mask = (r >= 1.5) & (r <= 5.0)
        r_pk = None
        if pk_mask.any():
            seg = g[pk_mask]
            i_pk_rel = int(np.argmax(seg))
            if seg[i_pk_rel] > max(2.0 * seg.mean(), 0.01):
                i_pk = i_pk_rel + np.flatnonzero(pk_mask)[0]
                r_pk = float(r[i_pk])
                v_mask = (r >= r_pk + 0.3) & (r <= r_pk + 1.5)
                if v_mask.any():
                    r_val = float(r[np.argmin(g[v_mask]) + np.flatnonzero(v_mask)[0]])

        # csv
        pair_tag = f"{el1}-{el2}"
        csv_path = outdir / f"{pair_tag}_rdf.csv"
        with open(csv_path, "w", encoding="utf-8") as fh:
            fh.write("r(A),g(r),CN(r)\n")
            for i in range(len(r)):
                fh.write(f"{r[i]:.5f},{g[i]:.6g},{cn[i]:.6g}\n")
        print(f"==> wrote {csv_path}")

        ax.plot(r, g, lw=1.0, label=f"g({el1}-{el2})")
        ax2 = ax.twinx()
        ax2.plot(r, cn, lw=0.8, ls="--", color="tab:red", label="CN")
        ax2.set_ylabel("CN(r)")
        if r_pk:
            ax.axvline(r_pk, color="tab:green", lw=0.7, ls=":")
        ax.set_xlabel("r (A)")
        ax.set_ylabel(f"g(r)  {el1}-{el2}")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)

        cn_s = f"{cn_at_cut:.2f}"
        if r_pk:
            conclusions.append(
                f"{el1}-{el2}: first peak at {r_pk:.2f} A (g={g[i_pk]:.2f}), "
                f"CN(<{args.cn_cut:.1f} A)={cn_s}")
        else:
            conclusions.append(f"{el1}-{el2}: no peak in [1.5,5.0] A, CN(<{args.cn_cut:.1f} A)={cn_s}")

    png = outdir / f"{args.name}_rdf_cn.png"
    savefig(fig, png)
    print_conclusion("; ".join(conclusions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
