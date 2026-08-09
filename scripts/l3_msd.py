#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""l3: MSD + diffusion coefficient (Einstein relation) -- pure numpy.

Ported and parameterized from the project's calc_msd.py (VASP AIMD analysis):
  - PBC unwrap: frac -> diff -> d -= round(d) -> cumsum (vectorized, idempotent
    on already-unwrapped input; see dump_io.unwrap_ortho)
  - MSD(t) = <|r(t) - r(0)|^2> with the first frame as reference; 3D total,
    xy in-plane and z components
  - D = slope * 1e-4 / dims  [cm^2/s]  (slope in A^2/ps, dims = 6/4/2),
    linear fit over the [--fit-start, --fit-end] fraction of the trajectory
  - optional COM drift subtraction (per-frame, reference-type COM; ported from
    MD-Analyzer _compute_msd, MIT)
  - --diffusive-check: log-log slope ~ 1 validates the Einstein regime

Input: prefers the l1_traj_prep product (<prefix>_uw.dump + <prefix>_meta.json,
from which dt_fs and the type->element map are read automatically); a raw dump
works too with --unwrap pbc and --timestep-fs.

Outputs: <name>_msd.csv (t_ps, msd_3d, msd_xy[, msd_z] per selection),
         <name>_msd.png, CONCLUSION line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_io import parse_dump, read_types_elements, unwrap_com, unwrap_ortho  # noqa: E402
from plot_utils import apply_style, print_conclusion, savefig  # noqa: E402
from region import add_region_args, select_traj  # noqa: E402


def msd_of(coords: np.ndarray, idx: np.ndarray, dims: str) -> dict:
    """MSD(t) for atom indices idx; dims in {'3d','xy','z'}.

    coords: (nframe, natom, 3) unwrapped. Returns {label: msd_array}.
    """
    x = coords[:, idx]
    disp = x - x[0]
    out = {}
    if dims in ("3d", "all"):
        out["msd_3d"] = np.mean(np.sum(disp ** 2, axis=2), axis=1)
    if dims in ("xy", "all"):
        out["msd_xy"] = np.mean(np.sum(disp[:, :, :2] ** 2, axis=2), axis=1)
    if dims in ("z", "all"):
        out["msd_z"] = np.mean(disp[:, :, 2] ** 2, axis=1)
    return out


def slope(tt: np.ndarray, mm: np.ndarray, lo: float, hi: float):
    """Linear slope of msd over the [lo, hi] fraction of the time axis."""
    m = (tt >= lo * tt[-1]) & (tt <= hi * tt[-1])
    return float(np.polyfit(tt[m], mm[m], 1)[0])


def diffusive_check(tt: np.ndarray, mm: np.ndarray, lo: float, hi: float) -> float:
    """log-log slope over the fit window; ~1 means Einstein regime."""
    m = (tt > 0) & (tt >= lo * tt[-1]) & (tt <= hi * tt[-1])
    return float(np.polyfit(np.log(tt[m]), np.log(mm[m]), 1)[0])


def load_meta(dump_path: str) -> dict | None:
    """Look for <prefix>_meta.json next to <prefix>_uw.dump and read dt_fs/map."""
    p = Path(dump_path)
    if p.suffix != ".dump":
        return None
    meta_path = p.with_name(p.stem.replace("_uw", "") + "_meta.json")
    if not meta_path.is_file():
        meta_path = p.with_suffix(".meta.json")
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="l3: MSD + diffusion coefficient (numpy)")
    ap.add_argument("--dump", required=True, help="<name>_uw.dump (l1 product) or raw dump")
    ap.add_argument("--data", default=None, help="data.lammps (type->element)")
    ap.add_argument("--unwrap", choices=["pbc", "none"], default="pbc",
                    help="unwrap wrapped input (no-op on xu/yu/zu dumps; "
                         "prefer l1_traj_prep output)")
    ap.add_argument("--timestep-fs", type=float, default=0.5,
                    help="MD timestep in fs (only for raw dumps without meta.json)")
    ap.add_argument("--types", default=None, help="type ids to analyse, e.g. '1 2'")
    ap.add_argument("--elements", default=None,
                    help="element symbols to analyse, e.g. 'O K' (requires type map)")
    ap.add_argument("--dims", choices=["3d", "xy", "z", "all"], default="all",
                    help="MSD components (default all: 3d, xy, z)")
    ap.add_argument("--fit-start", type=float, default=0.2,
                    help="fit window start as fraction of trajectory (default 0.2)")
    ap.add_argument("--fit-end", type=float, default=0.9,
                    help="fit window end as fraction (default 0.9)")
    ap.add_argument("--com-subtract", action="store_true",
                    help="remove per-frame COM drift of the analysed atoms "
                         "(requires unwrapped coords)")
    ap.add_argument("--diffusive-check", action="store_true",
                    help="report log-log MSD slope over the fit window")
    ap.add_argument("--elements-map", default=None, help="type->element override")
    ap.add_argument("--outdir", default=".", help="output directory")
    ap.add_argument("--name", default="msd", help="output file prefix")
    add_region_args(ap)
    args = ap.parse_args()

    if args.types is None and args.elements is None:
        raise SystemExit("ERROR: choose atoms with --types '1 2' or --elements 'O K'")
    if args.per_frame:
        raise SystemExit("ERROR: MSD forbids --per-frame (atom identity must stay "
                         "fixed across frames); use a static region")

    meta = load_meta(args.dump)
    if meta:
        print(f"    meta.json: dt_fs={meta.get('dt_fs')}, "
              f"unwrap={meta.get('unwrap_mode')}, source={meta.get('source_dump')}")

    d = parse_dump(args.dump, timestep_fs=args.timestep_fs)
    # dt_fs precedence: l1 meta.json > inferred from dump timestep gaps > CLI
    if meta and meta.get("dt_fs") is not None:
        dt_fs = meta["dt_fs"]
    elif d.dt_fs is not None:
        dt_fs = d.dt_fs
    else:
        dt_fs = args.timestep_fs
    if args.unwrap == "pbc" and d.wrapped:
        coords = unwrap_ortho(d.coords, d.cell)
        print("    PBC unwrap applied (raw wrapped input)")
    else:
        coords = d.coords

    # selection
    elements = read_types_elements(data_path=args.data, elements_map=args.elements_map)
    if not elements and meta:
        elements = {int(k): v for k, v in (meta.get("types_elements") or {}).items()}
    if args.elements:
        tset = set()
        for el in str(args.elements).split():
            hits = [t for t, e in elements.items() if e == el]
            if not hits:
                raise SystemExit(
                    f"ERROR: element '{el}' not in type map {elements}; "
                    "pass --data or --elements-map")
            tset.update(hits)
    else:
        tset = {int(t) for t in str(args.types).split()}
    mask_sel = np.isin(d.types[0], list(tset))
    if not mask_sel.any():
        raise SystemExit(f"ERROR: no atoms of types {sorted(tset)} in the trajectory")

    # region (static) AND selection
    mask = select_traj(coords, d.box, args, d.types)[0]
    keep = mask & mask_sel
    if not keep.any():
        raise SystemExit("ERROR: region+type selection is empty")
    print(f"    analysing {keep.sum()} atoms (types {sorted(tset)})")

    if args.com_subtract:
        coords = unwrap_com(coords, tset, d.types)

    # time axis
    t_frames = np.arange(len(d.timesteps), dtype=float)
    t_ps = t_frames * dt_fs / 1000.0  # fs -> ps

    selections = [{"label": "all", "idx": np.flatnonzero(keep)}]
    if args.elements and len(tset) > 1:
        # per-element breakdown when requested by element names
        for el in str(args.elements).split():
            et = [t for t, e in elements.items() if e == el and t in tset]
            emask = np.isin(d.types[0], et) & mask
            if emask.any():
                selections.append({"label": el, "idx": np.flatnonzero(emask)})
    elif args.types and len(tset) > 1:
        for t in sorted(tset):
            emask = (d.types[0] == t) & mask
            if emask.any():
                el = elements.get(t, f"T{t}")
                selections.append({"label": f"{el}", "idx": np.flatnonzero(emask)})

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    apply_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(7, 7),
                             gridspec_kw={"height_ratios": [3, 1]})
    ax_main, ax_log = axes

    csv_path = outdir / f"{args.name}_msd.csv"
    key_map = {"3d": "msd_3d", "xy": "msd_xy", "z": "msd_z"}
    key_list = key_map.values() if args.dims == "all" else [key_map[args.dims]]
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("t_ps," + ",".join(f"{s['label']}_{k}" for s in selections
                                    for k in key_list) + "\n")
        rows = []
        for sel in selections:
            rows.append(msd_of(coords, sel["idx"], args.dims))
        n = len(t_ps)
        for i in range(n):
            fh.write(f"{t_ps[i]:.6g}," + ",".join(
                f"{rows[j][k][i]:.6g}"
                for j in range(len(selections)) for k in key_list) + "\n")
    print(f"==> wrote {csv_path}")

    conclusions = []
    for si, sel in enumerate(selections):
        m = msd_of(coords, sel["idx"], args.dims)
        for dm, k in key_map.items():
            if k not in m:
                continue
            mm = m[k]
            s = slope(t_ps, mm, args.fit_start, args.fit_end)
            div = {"3d": 6.0, "xy": 4.0, "z": 2.0}[dm]
            dval = s * 1e-4 / div  # A^2/ps -> cm^2/s
            note = ""
            if args.diffusive_check:
                dlog = diffusive_check(t_ps, mm, args.fit_start, args.fit_end)
                note = f", log-log slope={dlog:.2f}"
                if abs(dlog - 1.0) > 0.2:
                    note += " (WARNING: not diffusive)"
            conclusions.append(
                f"D({sel['label']},{dm})={dval:.3e} cm2/s "
                f"({args.fit_start}-{args.fit_end} fit{note})")
            ax_main.plot(t_ps, mm, lw=1.0, label=f"{sel['label']} {dm}")
            # fit window highlight
            m_fit = (t_ps >= args.fit_start * t_ps[-1]) & (t_ps <= args.fit_end * t_ps[-1])
            ax_main.plot(t_ps[m_fit], mm[m_fit], lw=2.5, alpha=0.5)
            ax_log.loglog(t_ps[1:], mm[1:], lw=0.8, alpha=0.7,
                          label=f"{sel['label']} {dm}")

    ax_main.set_xlabel("t (ps)")
    ax_main.set_ylabel("MSD (A$^2$)")
    ax_main.legend(fontsize=8, ncol=2)
    ax_log.set_xlabel("t (ps, log)")
    ax_log.set_ylabel("MSD (log)")
    ax_log.legend(fontsize=7, ncol=2)
    png = outdir / f"{args.name}_msd.png"
    savefig(fig, png)

    print_conclusion("; ".join(conclusions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
