#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""l2: hydrogen-bond statistics + lifetime C(t) via MDAnalysis.

HydrogenBondAnalysis (MDAnalysis.analysis.hydrogenbonds; the old
MDAnalysis.analysis.hbonds path is removed in MDAnalysis 3.x) with the
classic water criteria (D-A cutoff 3.5 A, D-H-A angle 150 deg; overridable).

C(t) is the continuous autocorrelation, ported from MD-Analyzer (MIT, see
references/md_analyzer_notes.md #5): every (donor, acceptor) pair's frames are
split into consecutive runs and C(t) = sum over runs of max(L - t, 0) / sum L,
i.e. the probability that a bond present at t=0 survives uninterrupted to t.
Lifetime tau from a log-linear single-exponential fit (or double-exponential
via scipy curve_fit with bounds; tau_avg = A-weighted).

Region filtering: applied post-hoc to the hbond event table by donor/acceptor
atom ids (static first-frame mask). Optional --classify-z 'lo hi' labels
events interface/bulk/transition by donor/acceptor z (Z-criterion port, #7).

Outputs: <name>_hbonds.csv (frame, n_hbonds), <name>_hbond_ct.csv,
         <name>_hbonds.png, <name>_hbond_ct.png
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


def extract_runs(frames):
    """Split a sorted list of frames into consecutive runs."""
    frames = sorted(frames)
    runs = []
    start = prev = frames[0]
    for f in frames[1:]:
        if f == prev + 1:
            prev = f
        else:
            runs.append((start, prev))
            start = prev = f
    runs.append((start, prev))
    return [(hi - lo + 1) for lo, hi in runs]  # run lengths


def compute_ct(pair_frames: dict, n_frames: int) -> tuple[np.ndarray, int]:
    """C(t) from per-pair frame lists (continuous autocorrelation).

    Ported from MD-Analyzer analyzers/hbond/engine.py::_compute_ct (MIT).
    Returns (C(t), n_runs).
    """
    surv = np.zeros(n_frames, dtype=float)
    total = 0
    n_runs = 0
    for frames in pair_frames.values():
        for L in extract_runs(frames):
            L = min(int(L), n_frames)
            surv[:L] += np.arange(L, 0, -1, dtype=float)
            total += L
            n_runs += 1
    if total == 0:
        return np.zeros(n_frames), 0
    return surv / total, n_runs


def fit_lifetime(t, ct, double: bool = False) -> dict:
    """Fit tau from C(t). Single: log-linear, tau = -1/b over ct in (0.01, 1.0].
    Double: scipy curve_fit with bounds; tau_avg = A-weighted mean.
    Ported from MD-Analyzer _fit_lifetime (MIT).
    """
    m = (ct > 0.01) & (ct <= 1.0)
    if m.sum() < 3:
        return {"method": "none", "tau": None}
    tf, cf = t[m], ct[m]
    xm, ym = tf.mean(), np.log(cf).mean()
    b = np.sum((tf - xm) * (np.log(cf) - ym)) / np.sum((tf - xm) ** 2)
    if b >= 0:
        return {"method": "single", "tau": None, "note": "positive slope; no decay"}
    tau1 = -1.0 / b
    if not double:
        return {"method": "single", "tau": float(tau1)}
    from scipy.optimize import curve_fit

    def biexp(x, A1, tau_a, A2, tau_b):
        return A1 * np.exp(-x / tau_a) + A2 * np.exp(-x / tau_b)

    p0 = [0.7, 0.3 * tau1, 0.3, 3.0 * tau1]
    try:
        popt, _ = curve_fit(biexp, tf, cf, p0=p0,
                            bounds=([0.01, 0.1, 0.01, 0.1], [1.5, 1000, 1.5, 1000]),
                            maxfev=20000)
        A1, t1, A2, t2 = popt
        if t1 < t2:
            A1, t1, A2, t2 = A2, t2, A1, t1  # t1 = long component
        tau_avg = (A1 * t1 + A2 * t2) / (A1 + A2)
        return {"method": "double", "tau": float(tau_avg),
                "tau_long": float(t1), "tau_short": float(t2),
                "A_long": float(A1), "A_short": float(A2)}
    except Exception as exc:  # pragma: no cover - fit robustness
        return {"method": "single", "tau": float(tau1), "note": f"double fit failed: {exc}"}


def _check_pairing_consistency(u, by_name: dict, donors: str, hydrogens: str,
                               oh_cutoff: float = 1.3, frac_ok: float = 0.9,
                               n_sample: int = 8,
                               start: int = 0, stop: int = -1) -> None:
    """O-H pairing sanity check on sampled frames (raw vs minimum-image).

    Two failure modes are detected:
    - low RAW fraction while the minimum-image fraction is high: the dump's
      unwrap is inconsistent (image-flag drift in xu/yu/zu, or the input is
      wrapped). HydrogenBondAnalysis then silently loses donor-H pairs whose
      minimum-image distance collapses to ~0 (its min_cutoff=1.0 discards
      them), producing a fake H-bond *decline* over the trajectory (measured:
      HBA 160->100 while the PBC-correct count stays ~162).
    - low minimum-image fraction: coordinates themselves are corrupted.
    Warns only; analysis proceeds.
    """
    if not by_name.get(donors) or not by_name.get(hydrogens):
        return
    o_types = by_name[donors]
    h_types = by_name[hydrogens]
    nframe = len(u.trajectory)
    if nframe == 0:
        return
    # sample within the ACTUAL analysis window (start/stop), not the start
    # of the trajectory -- unwrap drift grows with time, so checking the
    # first frames would miss it
    stop = nframe if (stop is None or stop < 0) else min(stop, nframe)
    hi = max(start, stop - 1)
    frames = np.linspace(start, hi, min(n_sample, max(1, stop - start))).astype(int)
    frac_raw, frac_mic = [], []
    for fi in frames:
        u.trajectory[fi]
        pos = u.atoms.positions
        L = u.dimensions[:3]
        type_ids = u.atoms.types.astype(int)  # MDAnalysis stores types as strings
        o_idx = np.flatnonzero(np.isin(type_ids, o_types))
        h_idx = np.flatnonzero(np.isin(type_ids, h_types))
        if o_idx.size == 0 or h_idx.size == 0:
            continue
        dv = pos[o_idx][:, None, :] - pos[h_idx][None, :, :]
        d_raw = np.linalg.norm(dv, axis=2)
        dv = dv - L[None, None, :] * np.rint(dv / L[None, None, :])
        d_mic = np.linalg.norm(dv, axis=2)
        ok = lambda d: float(np.all(np.sort(d, axis=1)[:, :2] < oh_cutoff, axis=1).mean())
        frac_raw.append(ok(d_raw))
        frac_mic.append(ok(d_mic))
    if not frac_raw:
        return
    m_raw, m_mic = float(np.mean(frac_raw)), float(np.mean(frac_mic))
    if m_mic < frac_ok:
        print(
            f"WARNING: pairing check found only {m_mic * 100:.0f}% of {donors} "
            f"atoms with 2 {hydrogens} within {oh_cutoff} A even under minimum "
            f"image ({len(frames)} sampled frames) - coordinates look corrupted; "
            f"H-bond counts will not be reliable.")
    elif m_raw < frac_ok:
        print(
            f"WARNING: pairing check: {m_raw * 100:.0f}% of {donors} atoms have "
            f"2 {hydrogens} within {oh_cutoff} A in RAW coordinates "
            f"({m_mic * 100:.0f}% under minimum image) - the dump's unwrap is "
            f"inconsistent (image-flag drift in xu/yu/zu) or the input is "
            f"wrapped. HBA can then show a fake H-bond decline; verify with "
            f"a wrapped dump or --method mic.")


def run_hbonds_mic(u, o_types, h_types, start, stop, stride,
                   dc: float = 3.5, ang_deg: float = 150.0,
                   dhc: float = 1.2) -> tuple[np.ndarray, dict]:
    """PBC-correct H-bond event counting (pure numpy, minimum-image).

    Replaces HBA as the default: HBA's capped_distance uses min_cutoff=1.0,
    which discards donor-H pairs whose minimum-image distance collapses to
    ~0 -- exactly what happens for a molecule crossing the PBC boundary in
    any per-atom unwrap (LAMMPS xu/yu/zu or l1_traj_prep). As more molecules
    cross, HBA silently loses donors and reports a FAKE H-bond decline
    (measured: 160->100 while the true count stays ~162). The minimum-image
    count below has no such cutoff: a drifted pair is simply folded back.

    Every H is assigned to its nearest O within dhc (min-image), then each
    donor-H pair counts acceptor O's with D-A < dc (min-image) and
    D-H-A angle > ang_deg.

    Returns (n_per_frame, pair_frames) with pair_frames[(donor, acceptor)]
    -> list of RELATIVE frame indices, compatible with compute_ct().
    """
    n_frames = len(range(start, stop if stop else len(u.trajectory), stride))
    n_per_frame = np.zeros(n_frames, dtype=int)
    pair_frames: dict[tuple[int, int], list[int]] = {}
    cos_cut = np.cos(np.radians(ang_deg))
    k = 0
    types = u.atoms.types.astype(int)  # MDAnalysis stores types as strings
    for fi in range(start, stop if stop else len(u.trajectory), stride):
        u.trajectory[fi]
        pos = u.atoms.positions
        L = u.dimensions[:3]
        o_idx = np.flatnonzero(np.isin(types, o_types))
        h_idx = np.flatnonzero(np.isin(types, h_types))
        o = pos[o_idx]
        h = pos[h_idx]
        if o.size == 0 or h.size == 0:
            k += 1
            continue
        dv = o[:, None, :] - h[None, :, :]                 # (n_o, n_h, 3)
        dv = dv - L[None, None, :] * np.rint(dv / L[None, None, :])
        dd = np.linalg.norm(dv, axis=2)                    # (n_o, n_h)
        donor_of_h = np.argmin(dd, axis=0)
        h_ok = dd[donor_of_h, np.arange(h_idx.size)] < dhc
        hh = np.flatnonzero(h_ok)
        donors = donor_of_h[hh]
        # donor-H vectors in the O->H direction (note: dv is H->O; negate)
        v_dh = -dv[donors, hh]                             # (m, 3), O->H
        v_dh = v_dh - L[None, :] * np.rint(v_dh / L[None, :])
        # per-frame loop over donor-H pairs; acceptors vectorized
        for m in range(hh.size):
            d_i = int(donors[m])
            v_da = o - o[d_i]
            v_da = v_da - L[None, :] * np.rint(v_da / L[None, :])
            da = np.linalg.norm(v_da, axis=1)
            acc = np.flatnonzero((da < dc) & (np.arange(o.shape[0]) != d_i))
            if acc.size == 0:
                continue
            v_ha = o[acc] - h[hh[m]]
            v_ha = v_ha - L[None, :] * np.rint(v_ha / L[None, :])
            cosang = -np.sum(v_dh[m][None, :] * v_ha, axis=1) / (
                np.linalg.norm(v_dh[m]) * np.linalg.norm(v_ha, axis=1))
            # angle > ang_deg  <=>  cosang < cos(ang_deg)  (arccos is
            # decreasing); cosang IS cos(D-H-A), so accept the shallow ones
            # only when they are BELOW the cutoff
            for ai in np.flatnonzero(cosang < cos_cut):
                a_atom = int(o_idx[acc[ai]])
                d_atom = int(o_idx[d_i])
                n_per_frame[k] += 1
                pair_frames.setdefault((d_atom, a_atom), []).append(k)
        k += 1
    return n_per_frame, pair_frames


def main() -> int:
    ap = argparse.ArgumentParser(description="l2: hydrogen-bond stats + lifetime (MDAnalysis)")
    ap.add_argument("--dump", required=True, help="dump.lammpstrj or <name>_uw.dump")
    ap.add_argument("--data", default=None, help="data.lammps (type->element)")
    ap.add_argument("--start", type=int, default=0, help="first frame (default 0)")
    ap.add_argument("--stop", type=int, default=None,
                    help="last frame, exclusive (default: all)")
    ap.add_argument("--stride", type=int, default=1, help="frame stride (default 1)")
    ap.add_argument("--method", choices=["mic", "hba"], default="mic",
                    help="H-bond engine: 'mic' = pure-numpy minimum-image "
                         "(default; PBC-robust, no fake-decline artifact); "
                         "'hba' = MDAnalysis HydrogenBondAnalysis (legacy; "
                         "min_cutoff=1.0 discards boundary-crossing donor-H "
                         "pairs and can fake a decline on unwrapped dumps)")
    ap.add_argument("--donors", default="O", help="donor element (default O)")
    ap.add_argument("--acceptors", default="O", help="acceptor element (default O)")
    ap.add_argument("--hydrogens", default="H", help="hydrogen element (default H)")
    ap.add_argument("--distance", type=float, default=3.5,
                    help="donor-acceptor cutoff, A (default 3.5, water convention)")
    ap.add_argument("--angle", type=float, default=150.0,
                    help="donor-H-acceptor angle cutoff, deg (default 150)")
    ap.add_argument("--lifetime-method", choices=["single", "double"], default="single",
                    help="C(t) lifetime fit (default single, log-linear)")
    ap.add_argument("--classify-z", default=None, metavar="'LO HI'",
                    help="bulk z band; events outside are interface, crossing are "
                         "transition (optional)")
    ap.add_argument("--elements-map", default=None, help="type->element override")
    ap.add_argument("--outdir", default=".", help="output directory")
    ap.add_argument("--name", default="hbond", help="output file prefix")
    add_region_args(ap)
    args = ap.parse_args()

    import MDAnalysis as mda
    if args.data:
        u = mda.Universe(args.data, format="DATA")
        u.load_new(args.dump, format="LAMMPSDUMP")
    else:
        u = mda.Universe(args.dump, format="LAMMPSDUMP")

    elements = read_types_elements(data_path=args.data, elements_map=args.elements_map)
    by_name = {}
    for t, el in elements.items():
        by_name.setdefault(el, []).append(t)
    for which, el in (("donor", args.donors), ("acceptor", args.acceptors),
                      ("hydrogen", args.hydrogens)):
        if el not in by_name:
            raise SystemExit(
                f"ERROR: {which} element '{el}' not in type map {elements}; "
                "use --elements-map '1:O 2:H ...' or provide data.lammps Masses "
                "'# element' comments")

    # MDAnalysis stores atom types as strings ('1', '2', ...); cast to int
    type_ids = u.atoms.types.astype(int)
    n_donors = sum(int(np.sum(type_ids == t)) for t in by_name[args.donors])
    n_h = sum(int(np.sum(type_ids == t)) for t in by_name[args.hydrogens])
    n_acc = sum(int(np.sum(type_ids == t)) for t in by_name[args.acceptors])
    n_frames = len(range(args.start, args.stop if args.stop else len(u.trajectory),
                         args.stride))

    if args.method == "mic":
        type_ids = u.atoms.types.astype(int)
        print(f"==> MIC: {n_donors} donors, {n_h} hydrogens, {n_acc} acceptors, "
              f"d_a<{args.distance} A, angle>{args.angle} deg (PBC minimum-image)")
        n_per_frame, pair_frames = run_hbonds_mic(
            u, by_name[args.donors], by_name[args.hydrogens],
            args.start, args.stop, args.stride,
            dc=args.distance, ang_deg=args.angle)
        # light event table (frame_rel, donor, acceptor) for region/classify
        hb = None
    else:
        from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
        # MDAnalysis 2.10 HydrogenBondAnalysis requires selection strings
        sel_d = "type " + " ".join(map(str, by_name[args.donors]))
        sel_a = "type " + " ".join(map(str, by_name[args.acceptors]))
        sel_h = "type " + " ".join(map(str, by_name[args.hydrogens]))
        hba = HydrogenBondAnalysis(u, donors_sel=sel_d, hydrogens_sel=sel_h,
                                   acceptors_sel=sel_a, d_a_cutoff=args.distance,
                                   d_h_a_angle_cutoff=args.angle,
                                   update_selections=False)
        print(f"==> HBA: {n_donors} donors, {n_h} hydrogens, {n_acc} acceptors, "
              f"d_a<{args.distance} A, angle>{args.angle} deg")
        _check_pairing_consistency(u, by_name, args.donors, args.hydrogens,
                                   start=args.start, stop=args.stop)
        hba.run(start=args.start, stop=args.stop, step=args.stride, verbose=False)
        hb = hba.hbonds  # (n, 6): frame, donor, hydrogen, acceptor, dist, angle
        # HBA frame column is the ABSOLUTE trajectory index; map to relative
        # index within [start:stop:stride] so per-frame arrays (sized
        # n_frames) index correctly when start != 0.
        if len(hb):
            hb[:, 0] = (hb[:, 0].astype(int) - args.start) // max(1, args.stride)

        # per-frame bond count + pair table from the event table
        n_per_frame = np.zeros(n_frames, dtype=int)
        pair_frames: dict[tuple[int, int], list[int]] = {}
        for row in hb:
            f, d, a = int(row[0]), int(row[1]), int(row[3])
            n_per_frame[f] += 1
            pair_frames.setdefault((d, a), []).append(f)

    # region mask on atom indices (static first frame). Events carry 0-based
    # universe indices (topology order), not LAMMPS ids.
    u.trajectory[0]
    dims = u.dimensions[:3]
    args_box = np.array([[0, dims[0]], [0, dims[1]], [0, dims[2]]])
    mask = np.ones(len(u.atoms), dtype=bool)
    if args.region != "full" or args.layer_auto:
        mask = select_frame(u.atoms.positions, args_box, args,
                            u.atoms.types.astype(int))
        if not mask.any():
            raise SystemExit("ERROR: region mask selects zero atoms")
    inds_keep = set(np.flatnonzero(mask).tolist())
    if args.region != "full" or args.layer_auto:
        pair_frames = {(d, a): fl for (d, a), fl in pair_frames.items()
                       if d in inds_keep and a in inds_keep}
        # rebuild per-frame counts from the filtered pair table
        n_per_frame = np.zeros(n_frames, dtype=int)
        for fl in pair_frames.values():
            for f in fl:
                n_per_frame[f] += 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # csv: per-frame counts
    csv_hb = outdir / f"{args.name}_hbonds.csv"
    with open(csv_hb, "w", encoding="utf-8") as fh:
        fh.write("frame,n_hbonds\n")
        for f in range(n_frames):
            fh.write(f"{f},{n_per_frame[f]}\n")
    print(f"==> wrote {csv_hb}")

    # csv: C(t)
    ct, n_runs = compute_ct(pair_frames, n_frames)
    csv_ct = outdir / f"{args.name}_hbond_ct.csv"
    with open(csv_ct, "w", encoding="utf-8") as fh:
        fh.write("frame,C(t)\n")
        for f in range(n_frames):
            fh.write(f"{f},{ct[f]:.6g}\n")
    print(f"==> wrote {csv_ct}")

    # lifetime
    t = np.arange(n_frames, dtype=float)
    fit = fit_lifetime(t, ct, double=(args.lifetime_method == "double"))
    mean_hb = float(n_per_frame.mean())
    per_donor = mean_hb / n_donors

    # classify-z (optional)
    classify_txt = ""
    if args.classify_z:
        lo, hi = (float(x) for x in str(args.classify_z).split())
        u.trajectory[0]
        z_of = {i: float(a.position[2]) for i, a in enumerate(u.atoms)}
        counts = {"bulk": 0, "interface": 0, "transition": 0}
        for (d, a), fl in pair_frames.items():
            z_d, z_a = z_of.get(d, lo), z_of.get(a, lo)
            for _ in fl:
                if (lo <= z_d <= hi) and (lo <= z_a <= hi):
                    counts["bulk"] += 1
                elif (z_d < lo and z_a > hi) or (z_d > hi and z_a < lo):
                    counts["transition"] += 1
                else:
                    counts["interface"] += 1
        classify_txt = (f"; events bulk={counts['bulk']} interface={counts['interface']} "
                        f"transition={counts['transition']}")

    # plots
    apply_style()
    import matplotlib.pyplot as plt
    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.hist(n_per_frame, bins=min(30, int(n_per_frame.max() - n_per_frame.min()) + 1),
             alpha=0.7, color="tab:blue")
    ax1.set_xlabel("n_hbonds per frame")
    ax1.set_ylabel("frames")
    png1 = outdir / f"{args.name}_hbonds.png"
    savefig(fig1, png1)

    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    ax2.plot(t, ct, lw=1.2)
    if fit.get("tau"):
        ax2.axvline(fit["tau"], color="tab:red", lw=0.8, ls="--",
                    label=f"tau={fit['tau']:.2f} ({fit['method']})")
    ax2.set_xlabel("time (frames)")
    ax2.set_ylabel("C(t)")
    ax2.legend(fontsize=9)
    png2 = outdir / f"{args.name}_hbond_ct.png"
    savefig(fig2, png2)

    # time series of the per-frame H-bond count (reveals drift/trends that the
    # mean hides, e.g. an unwrap-artifact decline or a genuine restructuring)
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(t, n_per_frame, color="tab:blue", lw=0.5, alpha=0.35,
             label="per-frame")
    if n_frames > 20:
        win = min(41, max(5, n_frames // 20))
        if win % 2 == 0:
            win += 1
        k = np.ones(win) / win
        sm = np.convolve(n_per_frame, k, mode="same")
        sm[: win // 2] = np.nan
        sm[-(win // 2):] = np.nan
        ax3.plot(t, sm, color="tab:red", lw=2.0, label="rolling mean")
    ax3.set_xlabel("time (frames)")
    ax3.set_ylabel("n_hbonds")
    ax3.legend(fontsize=9)
    png3 = outdir / f"{args.name}_hbonds_timeseries.png"
    savefig(fig3, png3)

    tau_s = f"{fit['tau']:.2f}" if fit.get("tau") else "n/a"
    print_conclusion(
        f"mean {mean_hb:.1f} H-bonds/frame ({per_donor:.2f} per donor), "
        f"tau_hb={tau_s} frames ({fit['method']}, n_runs={n_runs}){classify_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
