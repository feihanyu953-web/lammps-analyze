#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""l2: solvation-shell hydrogen-bond classification around a moving ion group.

Companion to l2_hbonds (which reports GLOBAL water hbonds): this module classifies
every directed hbond (donor O-H...acceptor O) by whether the donor/acceptor O
sits inside the FIRST SOLVATION SHELL of a moving ion species -- shell membership
is re-evaluated every frame (MIC), so the region FOLLOWS diffusing ions, unlike
the static --region layer/sphere of other modules.

Subsets:
  shell_shell : donor AND acceptor O both in some ion's shell at that frame
  donor_shell : donor O in a shell (acceptor any water)
  bulk_bulk   : donor and acceptor both outside all shells

Outputs (prefix --name):
  <name>_hbshell_counts.csv  frame,n_shell,hb_per_shell,hb_per_bulk
  <name>_hbshell_ct.csv      frame,C_shell_shell,C_donor_shell,C_bulk_bulk
  <name>_hbshell_counts.png / <name>_hbshell_ct.png
CONCLUSION lines report the shell-vs-bulk hbond count ratio and per-subset tau.

Caveats: pick --shell-rcut at the ion-solvent RDF first minimum (l2_rdf_cn);
C(t) resolution is bounded by the frame spacing; small shell_shell event counts
can drop below the lifetime fit threshold (tau reported as n/a -- do not invent).
Time axis is the frame index; anchor physical time via Dump.timesteps_rel / dt.

Reuses the PBC-robust minimum-image engine, C(t) and lifetime fitting from
l2_hbonds (run_hbonds_mic / compute_ct / fit_lifetime).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_io import read_types_elements  # noqa: E402
from l2_hbonds import compute_ct, fit_lifetime, run_hbonds_mic  # noqa: E402
from plot_utils import apply_style, print_conclusion, savefig, smooth  # noqa: E402
from region import add_region_args, select_frame  # noqa: E402

SUBSETS = ("shell_shell", "donor_shell", "bulk_bulk")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="l2: hbond classification around a moving ion solvation shell")
    ap.add_argument("--dump", required=True, help="dump.lammpstrj or <name>_uw.dump")
    ap.add_argument("--data", default=None, help="data.lammps (type->element)")
    ap.add_argument("--shell-around", required=True, metavar="ELEMENT",
                    help="ion element defining the shell (e.g. K); shells are "
                         "re-computed every frame around ALL its atoms (MIC)")
    ap.add_argument("--shell-rcut", type=float, required=True, metavar="R",
                    help="ion-O first-shell cutoff, A -- take from the ion-O RDF "
                         "first minimum (l2_rdf_cn)")
    ap.add_argument("--oh-max", type=float, default=1.3,
                    help="O-H pairing distance for water identification, A")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None, help="exclusive (default all)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--donors", default="O")
    ap.add_argument("--acceptors", default="O")
    ap.add_argument("--hydrogens", default="H")
    ap.add_argument("--distance", type=float, default=3.5, help="D...A cutoff, A")
    ap.add_argument("--angle", type=float, default=150.0, help="D-H...A cutoff, deg")
    ap.add_argument("--lifetime-method", choices=["single", "double"], default="single")
    ap.add_argument("--roll", type=int, default=41, help="rolling window for the "
                   "counts plot (edge-safe smooth; 0 disables)")
    ap.add_argument("--elements-map", default=None, help="type->element override")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--name", default="hbshell")
    add_region_args(ap)
    args = ap.parse_args()

    import MDAnalysis as mda
    if args.data:
        u = mda.Universe(args.data, format="DATA")
        u.load_new(args.dump, format="LAMMPSDUMP")
    else:
        u = mda.Universe(args.dump, format="LAMMPSDUMP")

    elements = read_types_elements(data_path=args.data, elements_map=args.elements_map)
    by_name: dict[str, list[int]] = {}
    for t, el in elements.items():
        by_name.setdefault(el, []).append(t)
    need = {"donor": args.donors, "acceptor": args.acceptors,
            "hydrogen": args.hydrogens, "shell ion": args.shell_around}
    for which, el in need.items():
        if el not in by_name:
            raise SystemExit(f"ERROR: {which} element '{el}' not in type map "
                             f"{elements}; pass --elements-map '1:O 2:H ...'")

    # static region mask (first frame): restricts BOTH the shell ions and the
    # hbond events (same post-hoc convention as l2_hbonds)
    u.trajectory[0]
    dims = u.dimensions[:3]
    args_box = np.array([[0, dims[0]], [0, dims[1]], [0, dims[2]]])
    mask = np.ones(len(u.atoms), dtype=bool)
    if args.region != "full" or args.layer_auto:
        mask = select_frame(u.atoms.positions, args_box, args,
                            u.atoms.types.astype(int))
        if not mask.any():
            raise SystemExit("ERROR: region mask selects zero atoms")

    type_ids = u.atoms.types.astype(int)
    k_idx_all = np.flatnonzero(np.isin(type_ids, by_name[args.shell_around]))
    k_idx = k_idx_all[mask[k_idx_all]]
    o_idx_all = np.flatnonzero(np.isin(type_ids, by_name[args.donors]))
    o_idx = o_idx_all[mask[o_idx_all]]
    if k_idx.size == 0 or o_idx.size == 0:
        raise SystemExit("ERROR: no shell ions / donors inside the region")

    n_frames = len(range(args.start, args.stop if args.stop else len(u.trajectory),
                         args.stride))
    print(f"==> MIC: {k_idx.size} '{args.shell_around}' ions, "
          f"shell rcut={args.shell_rcut} A; hbonds d_a<{args.distance} A, "
          f"angle>{args.angle} deg")

    # 1) global hbond events (donor/acceptor = universe atom indices)
    _, pair_frames = run_hbonds_mic(
        u, by_name[args.donors], by_name[args.hydrogens],
        args.start, args.stop, args.stride,
        dc=args.distance, ang_deg=args.angle)
    if args.region != "full" or args.layer_auto:
        keep = set(np.flatnonzero(mask).tolist())
        pair_frames = {(d, a): fl for (d, a), fl in pair_frames.items()
                       if d in keep and a in keep}

    # 2) per-frame shell membership over O atom indices (MIC, follows ions)
    o_pos_of = {int(i): n for n, i in enumerate(o_idx)}  # atom idx -> O-array idx
    frame_events: list[list[tuple[int, int]]] = [[] for _ in range(n_frames)]
    for (d, a), fl in pair_frames.items():
        for f in fl:
            frame_events[f].append((d, a))
    shell_sets: list[set[int]] = []
    n_shell_per_frame = np.zeros(n_frames, dtype=int)
    hb_per_shell = np.zeros(n_frames)
    hb_per_bulk = np.zeros(n_frames)
    for k, fi in enumerate(range(args.start, args.stop if args.stop
                                 else len(u.trajectory), args.stride)):
        u.trajectory[fi]
        pos = u.atoms.positions
        L = u.dimensions[:3]
        ok = pos[o_idx]
        kk = pos[k_idx]
        dv = ok[None, :, :] - kk[:, None, :]              # (n_k, n_o, 3)
        dv = dv - L[None, None, :] * np.rint(dv / L[None, None, :])
        in_shell = (np.linalg.norm(dv, axis=2) < args.shell_rcut).any(axis=0)
        shell_atoms = {int(o_idx[j]) for j in np.flatnonzero(in_shell)}
        shell_sets.append(shell_atoms)
        n_shell_per_frame[k] = len(shell_atoms)
        touch: dict[int, int] = {}
        for d, a in frame_events[k]:
            touch[d] = touch.get(d, 0) + 1
            touch[a] = touch.get(a, 0) + 1
        sh_vals = [touch.get(i, 0) for i in shell_atoms]
        bu_vals = [touch.get(int(i), 0)
                   for i in o_idx if int(i) not in shell_atoms]
        hb_per_shell[k] = float(np.mean(sh_vals)) if sh_vals else np.nan
        hb_per_bulk[k] = float(np.mean(bu_vals)) if bu_vals else np.nan

    mean_shell_pop = float(n_shell_per_frame.mean())

    # 3) per-subset C(t): a (pair, frame) event is classified by shell
    #    membership AT THAT FRAME; a pair may contribute to several subsets
    subset_pairs: dict[str, dict] = {s: {} for s in SUBSETS}
    for (d, a), fl in pair_frames.items():
        for f in fl:
            d_in = d in shell_sets[f]
            a_in = a in shell_sets[f]
            if d_in and a_in:
                key = "shell_shell"
            elif d_in:
                key = "donor_shell"
            elif not d_in and not a_in:
                key = "bulk_bulk"
            else:
                continue  # acceptor_shell-only: donor-side classes stay honest
            subset_pairs[key].setdefault((d, a), []).append(f)

    ct = {}
    tau = {}
    n_runs = {}
    t = np.arange(n_frames, dtype=float)
    for s in SUBSETS:
        ct[s], n_runs[s] = compute_ct(subset_pairs[s], n_frames)
        fit = fit_lifetime(t, ct[s], double=(args.lifetime_method == "double"))
        tau[s] = fit.get("tau")

    # 4) outputs
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_c = outdir / f"{args.name}_hbshell_counts.csv"
    with open(csv_c, "w", encoding="utf-8") as fh:
        fh.write("frame,n_shell,hb_per_shell,hb_per_bulk\n")
        for f in range(n_frames):
            fh.write(f"{f},{n_shell_per_frame[f]},"
                     f"{hb_per_shell[f]:.4f},{hb_per_bulk[f]:.4f}\n")
    print(f"==> wrote {csv_c}")
    csv_l = outdir / f"{args.name}_hbshell_ct.csv"
    with open(csv_l, "w", encoding="utf-8") as fh:
        fh.write("frame,C_shell_shell,C_donor_shell,C_bulk_bulk\n")
        for f in range(n_frames):
            fh.write(f"{f},{ct['shell_shell'][f]:.6g},"
                     f"{ct['donor_shell'][f]:.6g},{ct['bulk_bulk'][f]:.6g}\n")
    print(f"==> wrote {csv_l}")

    apply_style()
    import matplotlib.pyplot as plt
    fig1, ax1 = plt.subplots(figsize=(7, 3.8))
    ax1.plot(t, hb_per_shell, lw=0.5, alpha=0.3, color="tab:blue")
    ax1.plot(t, hb_per_bulk, lw=0.5, alpha=0.3, color="tab:orange")
    if args.roll and n_frames > args.roll:
        ax1.plot(t, smooth(hb_per_shell, args.roll), lw=2,
                 color="tab:blue", label=f"shell (roll {args.roll})")
        ax1.plot(t, smooth(hb_per_bulk, args.roll), lw=2,
                 color="tab:orange", label="bulk")
    ax1.set_xlabel("frame")
    ax1.set_ylabel("hbonds per water")
    ax1.legend(fontsize=9)
    savefig(fig1, outdir / f"{args.name}_hbshell_counts.png")

    fig2, ax2 = plt.subplots(figsize=(6, 3.8))
    for s, c in zip(SUBSETS, ("tab:blue", "tab:green", "tab:orange")):
        lab = f"{s} (tau={tau[s]:.2f})" if tau[s] else f"{s} (tau=n/a)"
        ax2.semilogy(t, np.maximum(ct[s], 1e-6), lw=1.2, color=c, label=lab)
    ax2.set_xlabel("time (frames)")
    ax2.set_ylabel("C(t)")
    ax2.legend(fontsize=8)
    savefig(fig2, outdir / f"{args.name}_hbshell_ct.png")

    m_sh = float(np.nanmean(hb_per_shell))
    m_bu = float(np.nanmean(hb_per_bulk))
    ratio = m_sh / m_bu if m_bu else float("nan")
    tau_txt = ", ".join(
        f"{s}={'n/a' if tau[s] is None else f'{tau[s]:.2f}'}" for s in SUBSETS)
    print_conclusion(
        f"K-like shell '{args.shell_around}' rcut={args.shell_rcut} A: "
        f"shell waters carry {m_sh:.2f} hbonds/water vs bulk {m_bu:.2f} "
        f"(ratio {ratio:.2f}); mean shell population {mean_shell_pop:.1f}; "
        f"tau (frames, {args.lifetime_method}): {tau_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
