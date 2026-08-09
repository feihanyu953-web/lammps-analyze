#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""l1: trajectory preparation -- read dump, PBC-unwrap / COM-dedrift,
resample, region-filter, and write a native lammpstrj + meta.json.

Outputs:
  <name>_uw.dump      native lammpstrj ('id type xu yu zu'), OVITO-readable
  <name>_meta.json    dt_fs, natoms, n_frames, type->element, applied
                      start/stop/stride, unwrap mode, region parameters

Downstream: l3_msd.py reads the meta.json automatically (dt_fs, element map).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_io import (  # noqa: E402
    Dump,
    parse_dump,
    read_types_elements,
    unwrap_com,
    unwrap_ortho,
    write_unwrapped_dump,
)
from plot_utils import print_conclusion  # noqa: E402
from region import add_region_args, select_traj  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description="l1: LAMMPS dump preprocessing (unwrap / COM / resample / region)")
    ap.add_argument("--dump", required=True, help="dump.lammpstrj input")
    ap.add_argument("--data", default=None, help="data.lammps (type->element from Masses)")
    ap.add_argument("--start", type=int, default=0, help="first frame (default 0)")
    ap.add_argument("--stop", type=int, default=-1, help="last frame, exclusive (default all)")
    ap.add_argument("--stride", type=int, default=1, help="frame stride (default 1)")
    ap.add_argument("--timestep-fs", type=float, default=0.5,
                    help="MD timestep in fs, for dt_fs inference (default 0.5)")
    ap.add_argument("--unwrap", choices=["pbc", "com", "none"], default="pbc",
                    help="pbc: nearest-image unwrap; com: also remove reference-type "
                         "center-of-mass drift (requires --com-ref); none: as-is")
    ap.add_argument("--com-ref", default=None, metavar="TYPES",
                    help="space-separated type ids defining the COM reference, e.g. '1 2'")
    ap.add_argument("--elements-map", default=None,
                    help="type->element override, e.g. '1:H 2:O' (wins over --data)")
    ap.add_argument("--write-xyz", action="store_true",
                    help="additionally write <name>.xyz (frame 0 only)")
    ap.add_argument("--outdir", default=".", help="output directory")
    ap.add_argument("--name", default="traj", help="output file prefix")
    add_region_args(ap)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = outdir / args.name

    print(f"==> parsing {args.dump} (start={args.start} stop={args.stop} stride={args.stride})")
    d: Dump = parse_dump(args.dump, start=args.start, stop=args.stop,
                         stride=args.stride, timestep_fs=args.timestep_fs)
    print(f"    {d.natoms} atoms x {len(d.timesteps)} frames, "
          f"dt_fs={d.dt_fs}, wrapped={d.wrapped}")

    # --- region: static (first-frame) mask, applied to atoms kept downstream ---
    types = d.types
    if args.region != "full" or args.layer_auto:
        mask = select_traj(d.coords, d.box, args, types)[0]  # static
        if not mask.any():
            raise SystemExit("ERROR: region mask selects zero atoms")
        keep = np.where(mask)[0]
        coords = d.coords[:, keep]
        atom_ids = d.atom_ids[:, keep]
        types = d.types[:, keep]
        print(f"    region kept {keep.size}/{d.natoms} atoms "
              f"(ids {atom_ids[0, 0]}..{atom_ids[0, -1]})")
    else:
        coords, atom_ids = d.coords, d.atom_ids

    # --- unwrap ---
    unwrap_mode = args.unwrap
    if unwrap_mode == "pbc":
        if not d.wrapped:
            print("    input already unwrapped (xu/yu/zu); PBC unwrap is a no-op")
        else:
            coords = unwrap_ortho(coords, d.cell)
    elif unwrap_mode == "com":
        if args.com_ref is None:
            raise SystemExit("ERROR: --unwrap com requires --com-ref 'TYPES' "
                             "(space-separated type ids)")
        if d.wrapped:
            coords = unwrap_ortho(coords, d.cell)  # COM drift needs unwrapped coords
        ref_types = {int(t) for t in str(args.com_ref).split()}
        coords = unwrap_com(coords, ref_types, types)
        print(f"    COM drift removed using types {sorted(ref_types)}")

    # --- write outputs ---
    dump_out = f"{prefix}_uw.dump"
    write_unwrapped_dump(dump_out, d.timesteps, d.box, atom_ids, types, coords)
    print(f"==> wrote {dump_out}")

    elements = read_types_elements(data_path=args.data, elements_map=args.elements_map)
    meta = {
        "skill": "lammps-analyze/l1_traj_prep",
        "source_dump": str(args.dump),
        "natoms": int(d.natoms),
        "n_frames": int(len(d.timesteps)),
        "dt_fs": d.dt_fs,
        "timestep_fs": args.timestep_fs,
        "applied": {"start": args.start, "stop": args.stop, "stride": args.stride},
        "unwrap_mode": unwrap_mode,
        "com_ref_types": [int(t) for t in str(args.com_ref).split()] if args.com_ref else None,
        "wrapped_input": bool(d.wrapped),
        "region": {k: getattr(args, k) for k in
                   ("region", "box_min", "box_max", "center", "r_in", "r_out",
                    "layer", "trunc", "zmin", "zmax", "layer_auto",
                    "marker_type", "zone_eps", "zone_margin", "zone")},
        "types_elements": {str(k): v for k, v in sorted(elements.items())},
    }
    meta_path = f"{prefix}_meta.json"
    Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"==> wrote {meta_path}")

    if args.write_xyz:
        xyz = f"{prefix}.xyz"
        with open(xyz, "w", encoding="utf-8") as fh:
            fh.write(f"{coords.shape[1]}\nframe 0\n")
            for a in range(coords.shape[1]):
                el = elements.get(int(types[0, a]), f"T{int(types[0, a])}")
                x, y, z = coords[0, a]
                fh.write(f"{el} {x:.6f} {y:.6f} {z:.6f}\n")
        print(f"==> wrote {xyz}")

    print_conclusion(
        f"prepared {len(d.timesteps)} frames x {coords.shape[1]} atoms "
        f"(of {d.natoms}) from {Path(args.dump).name}, dt={d.dt_fs} fs, "
        f"unwrap={unwrap_mode}, output prefix {args.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
