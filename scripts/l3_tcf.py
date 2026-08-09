#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""l3: generic time-correlation functions for water -- residence and dipole
orientation relaxation (pure numpy + molecules.py).

Modes:
  residence  -- continuous residence TCF of water molecules in a z-region
                (interface layer). C(t) = fraction of molecules present at
                t0 that remain continuously inside until t0+t. tau from the
                single-exponential fit (reuses l2_hbonds.fit_lifetime logic).
  dipole     -- orientational relaxation of the molecular dipole:
                C(t) = <mu_hat(t0+t) . mu_hat(t0)>. tau = rotational
                relaxation time (single-exponential fit).

Both need water molecules (O-H pairing via molecules.py, so --data or
--elements-map with O/H types is required) and unwrapped coordinates
(prefer an l1_traj_prep product).

Outputs: <name>_tcf.csv (t_ps, C(t)), <name>_tcf.png, CONCLUSION line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_io import parse_dump, read_types_elements  # noqa: E402
from molecules import find_water_molecules, water_dipoles, water_o_positions  # noqa: E402
from plot_utils import apply_style, print_conclusion, savefig  # noqa: E402
from region import add_region_args, select_frame  # noqa: E402


def residence_tcf(in_region: np.ndarray) -> tuple[np.ndarray, int]:
    """Continuous residence TCF (Impey et al. 1983, ensemble-averaged).

    in_region: (nframe, n_mol) bool. C(t) = <h(t0) h(t0+1) ... h(t0+t)> /
    <h(t0)>, averaged over all time origins t0 -- i.e. the fraction of
    molecule-frame pairs present at t0 that remain continuously inside
    through t0+t. Molecules that never leave give C(t) = 1 for all t.
    Returns (C(t), n_molecules_ever_in).
    """
    nframe = in_region.shape[0]
    ever = in_region.any(axis=0)
    n_ever = int(ever.sum())
    if n_ever == 0:
        raise ValueError("no molecule ever enters the region")
    ct = np.zeros(nframe)
    for t in range(nframe):
        cnt = 0
        total = 0
        for t0 in range(nframe - t):
            starts = in_region[t0]
            total += int(starts.sum())
            seg = np.all(in_region[t0: t0 + t + 1], axis=0)
            cnt += int(seg[starts].sum())
        ct[t] = cnt / total if total else 0.0
    return ct, n_ever


def dipole_tcf(mu: np.ndarray) -> np.ndarray:
    """Orientational relaxation C(t) = <mu_hat(t+t0) . mu_hat(t0)> over t0."""
    nframe, n_mol, _ = mu.shape
    nrm = np.linalg.norm(mu, axis=2)
    muhat = mu / np.where(nrm > 0, nrm, 1.0)[:, :, None]
    ct = np.zeros(nframe)
    for t in range(nframe):
        ct[t] = float(np.mean(np.sum(muhat[t:] * muhat[: nframe - t], axis=2)))
    return ct


def fit_tau(t, ct) -> dict:
    """Single-exponential tau via log-linear fit (shared logic with l2_hbonds)."""
    m = (ct > 0.01) & (ct <= 1.0)
    if m.sum() < 3:
        return {"tau": None}
    tf, cf = t[m], ct[m]
    xm, ym = tf.mean(), np.log(cf).mean()
    b = np.sum((tf - xm) * (np.log(cf) - ym)) / np.sum((tf - xm) ** 2)
    if b >= 0:
        return {"tau": None}
    return {"tau": float(-1.0 / b)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="l3: residence / dipole-relaxation TCF for water")
    ap.add_argument("--dump", required=True, help="<name>_uw.dump (unwrapped)")
    ap.add_argument("--data", default=None, help="data.lammps (type->element)")
    ap.add_argument("--start", type=int, default=0, help="first frame (default 0)")
    ap.add_argument("--stop", type=int, default=-1, help="last frame, exclusive (default all)")
    ap.add_argument("--stride", type=int, default=1, help="frame stride (default 1)")
    ap.add_argument("--mode", choices=["residence", "dipole", "both"], default="both")
    ap.add_argument("--timestep-fs", type=float, default=0.5,
                    help="MD timestep in fs (used when meta/dt inference fails)")
    ap.add_argument("--elements-map", default=None, help="type->element override")
    ap.add_argument("--outdir", default=".", help="output directory")
    ap.add_argument("--name", default="tcf", help="output file prefix")
    add_region_args(ap)
    args = ap.parse_args()

    if args.per_frame:
        raise SystemExit("ERROR: TCF needs fixed atom identity; use a static region")

    d = parse_dump(args.dump, start=args.start, stop=args.stop, stride=args.stride,
                   timestep_fs=args.timestep_fs)
    if d.wrapped:
        raise SystemExit(
            "ERROR: TCF needs unwrapped coordinates; run l1_traj_prep --unwrap pbc first")

    elements = read_types_elements(data_path=args.data, elements_map=args.elements_map)
    o_types = [t for t, e in elements.items() if e == "O"]
    h_types = [t for t, e in elements.items() if e == "H"]
    if not o_types or not h_types:
        raise SystemExit(
            f"ERROR: need O and H types in the map (got {elements}); "
            "pass --data or --elements-map '1:O 2:H'")

    mols = find_water_molecules(d.types, d.coords, o_types[0], h_types)
    if mols is None:
        raise SystemExit("ERROR: could not pair O-H into water molecules "
                         "(check O/H types and that the dump has unwrapped coords)")
    print(f"    {mols.shape[1]} water molecules x {mols.shape[0]} frames")

    dt_ps = (d.dt_fs if d.dt_fs is not None else args.timestep_fs) / 1000.0
    t_ps = np.arange(len(d.timesteps)) * dt_ps

    results = {}
    if args.mode in ("residence", "both"):
        if args.region == "full" and not args.layer_auto:
            raise SystemExit(
                "ERROR: residence mode needs a region (e.g. --region layer "
                "--layer top --trunc 3.0, or --zmin/--zmax) to define the layer")
        o_pos = water_o_positions(mols, d.coords)
        args_box = None  # select_frame builds its own per call
        # region mask per molecule from its O position (static: frame 0 mask)
        # NB: for residence we evaluate the same geometric layer every frame
        nframe = o_pos.shape[0]
        in_region = np.zeros((nframe, mols.shape[1]), dtype=bool)
        # build a per-frame region decision using select_frame semantics
        for f in range(nframe):
            box = d.box[f]
            args_box = np.array([[box[0, 0], box[0, 1]], [box[1, 0], box[1, 1]],
                                 [box[2, 0], box[2, 1]]])
            if args.region == "layer" and args.layer_auto:
                raise SystemExit("residence + --layer-auto not supported; "
                                 "use --layer top|bottom + --trunc or --zmin/--zmax")
            m = select_frame(o_pos[f], args_box, args)
            in_region[f] = m
        ct, n_ever = residence_tcf(in_region)
        fit = fit_tau(t_ps, ct)
        results["residence"] = (ct, fit, n_ever)

    if args.mode in ("dipole", "both"):
        mu = water_dipoles(mols, d.coords)
        ct = dipole_tcf(mu)
        fit = fit_tau(t_ps, ct)
        results["dipole"] = (ct, fit, mols.shape[1])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / f"{args.name}_tcf.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        cols = list(results)
        fh.write("t_ps," + ",".join(cols) + "\n")
        n = len(t_ps)
        for i in range(n):
            fh.write(f"{t_ps[i]:.6g}," + ",".join(
                f"{results[k][0][i]:.6g}" for k in cols) + "\n")
    print(f"==> wrote {csv_path}")

    apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k in results:
        ct, fit, n_ = results[k]
        ax.plot(t_ps, ct, lw=1.2, label=k)
        if fit.get("tau"):
            ax.axvline(fit["tau"], lw=0.8, ls="--",
                       label=f"tau({k})={fit['tau']:.2f} ps")
    ax.set_xlabel("t (ps)")
    ax.set_ylabel("C(t)")
    ax.legend(fontsize=9)
    png = outdir / f"{args.name}_tcf.png"
    savefig(fig, png)

    parts = []
    for k in results:
        ct, fit, n_ = results[k]
        tau_s = f"{fit['tau']:.2f} ps" if fit.get("tau") else "n/a"
        parts.append(f"tau({k})={tau_s} (n={n_})")
    print_conclusion("; ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
