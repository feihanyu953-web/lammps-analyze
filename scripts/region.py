#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spatial region selection for lammps-analyze (pure numpy).

Regions: box / sphere / layer (relative to instantaneous box or absolute z) /
layer-auto (double-surface detection, ported from MD-Analyzer's _detect_zones,
MIT) / full.

Static (first-frame) masks are the default; --per-frame reselects per frame
(valid for RDF/density; MSD forbids it because atom identity must stay fixed).

Design notes (see references/md_analyzer_notes.md):
- The ported filter_atoms_spatial keeps a single parameter dict (the original
  had a key-name mismatch that silently degraded custom regions to full-box).
- _detect_zones: split marker atom z at the median into lower/upper surfaces,
  interface = surf +/- eps, bulk = between surfaces minus cutoff+margin.
"""

from __future__ import annotations

import argparse
import re

import numpy as np

_AUTO_ZONE_RE = re.compile(r"^(interface|bulk)$")


def add_region_args(ap: argparse.ArgumentParser) -> None:
    """Register the standard region arguments on an argparse parser."""
    grp = ap.add_argument_group("region")
    grp.add_argument("--region", choices=["full", "box", "sphere", "layer"],
                     default="full", help="spatial selection (default: full box)")
    grp.add_argument("--box-min", default=None, metavar="X Y Z",
                     help="box region lower corner, e.g. '10 0 0'")
    grp.add_argument("--box-max", default=None, metavar="X Y Z",
                     help="box region upper corner, e.g. '20 20 20'")
    grp.add_argument("--center", default=None, metavar="X Y Z",
                     help="sphere region center")
    grp.add_argument("--r-in", type=float, default=0.0, help="sphere inner radius (A)")
    grp.add_argument("--r-out", type=float, default=10.0, help="sphere outer radius (A)")
    grp.add_argument("--layer", choices=["top", "bottom"], default=None,
                     help="layer relative to instantaneous box edge")
    grp.add_argument("--trunc", type=float, default=0.0,
                     help="layer inset from the box edge (A)")
    grp.add_argument("--zmin", type=float, default=None,
                     help="absolute-z lower bound for a layer band")
    grp.add_argument("--zmax", type=float, default=None,
                     help="absolute-z upper bound for a layer band")
    grp.add_argument("--layer-auto", action="store_true",
                     help="auto-detect double surfaces from marker atoms")
    grp.add_argument("--marker-type", type=int, default=None,
                     help="type id of the marker atoms (surface atoms) for --layer-auto")
    grp.add_argument("--zone-eps", type=float, default=1.0,
                     help="interface half-thickness around each surface (A)")
    grp.add_argument("--zone-margin", type=float, default=3.0,
                     help="exclusion margin between interface and bulk zones (A)")
    grp.add_argument("--zone", choices=["interface", "bulk"], default="interface",
                     help="which auto-detected zone to select (--layer-auto)")
    grp.add_argument("--per-frame", action="store_true",
                     help="reselect atoms every frame (RDF/density only; MSD forbids)")


def _vec3(s, name):
    if s is None:
        return None
    try:
        return np.array([float(x) for x in s.split()])
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--{name} must be three numbers separated by spaces, got '{s}'")


def select_frame(coords: np.ndarray, box: np.ndarray, args,
                 types: np.ndarray | None = None) -> np.ndarray:
    """Mask for one frame.

    coords: (natom, 3) wrapped cartesian; box: (3, 2) lo/hi;
    types: (natom,) type ids, required for --layer-auto (marker atoms).
    Returns a boolean array of length natom.
    """
    natoms = coords.shape[0]
    if args.region == "full":
        return np.ones(natoms, dtype=bool)

    if args.region == "box":
        lo = _vec3(args.box_min, "box-min")
        hi = _vec3(args.box_max, "box-max")
        if lo is None or hi is None:
            raise ValueError("--region box requires --box-min and --box-max")
        mask = np.ones(natoms, dtype=bool)
        for d in range(3):
            mask &= (coords[:, d] >= lo[d]) & (coords[:, d] <= hi[d])
        return mask

    if args.region == "sphere":
        c = _vec3(args.center, "center")
        if c is None:
            raise ValueError("--region sphere requires --center")
        dist = np.linalg.norm(coords - c, axis=1)
        return (dist >= args.r_in) & (dist <= args.r_out)

    # layer
    if args.layer_auto:
        if args.marker_type is None:
            raise ValueError("--layer-auto requires --marker-type")
        if types is None:
            raise ValueError("--layer-auto requires per-atom types (internal: types=)")
        marker_z = coords[types == args.marker_type, 2]
        zones = detect_zones(marker_z, eps=args.zone_eps, margin=args.zone_margin)
        if zones is None:
            raise ValueError(
                "--layer-auto: could not detect two distinct surfaces from "
                "marker-type atoms; check --marker-type and trajectory")
        interface, bulk = zones
        z = coords[:, 2]  # select by z over ALL atoms, surfaces came from markers
        if args.zone == "bulk":
            return (z >= bulk[0]) & (z <= bulk[1])
        return ((z >= interface[0][0]) & (z <= interface[0][1])) | \
               ((z >= interface[1][0]) & (z <= interface[1][1]))

    if args.zmin is not None or args.zmax is not None:
        lo = args.zmin if args.zmin is not None else -np.inf
        hi = args.zmax if args.zmax is not None else np.inf
        return (coords[:, 2] >= lo) & (coords[:, 2] <= hi)

    if args.layer == "top":
        return coords[:, 2] >= box[2, 1] - args.trunc
    if args.layer == "bottom":
        return coords[:, 2] <= box[2, 0] + args.trunc
    raise ValueError(
        "--region layer needs --layer top|bottom (+ --trunc), --zmin/--zmax, "
        "or --layer-auto (+ --marker-type)")


def select_traj(coords: np.ndarray, box: np.ndarray, args,
                types: np.ndarray | None = None) -> np.ndarray:
    """Mask over the whole trajectory.

    Static (first-frame) mask by default; per-frame mask if --per-frame.
    Returns (nframe, natom) bool. types: (nframe, natom) or (natom,), required
    for --layer-auto.
    """
    nframe = coords.shape[0]
    if args.per_frame:
        return np.stack([select_frame(coords[f], box[f], args, types[f] if types is not None else None)
                         for f in range(nframe)])
    m0 = select_frame(coords[0], box[0], args,
                      types[0] if types is not None and types.ndim == 2 else types)
    return np.broadcast_to(m0, (nframe, m0.shape[0]))


def detect_zones(marker_z: np.ndarray, eps: float = 1.0, margin: float = 3.0):
    """Detect two slab surfaces from marker-atom z coordinates.

    Ported from MD-Analyzer analyzers/msd/engine.py::_detect_zones (MIT; see
    references/md_analyzer_notes.md #2). Splits marker z at the median into
    lower/upper surfaces, then:
      interface = [(surf_lower - eps, surf_lower + eps),
                   (surf_upper - eps, surf_upper + eps)]
      bulk      = (surf_lower + eps + margin, surf_upper - eps - margin)
    Returns None if a double-surface system is not detected (gap <= 0 or > 100 A).
    """
    if marker_z.size == 0:
        return None
    z_min, z_max = float(marker_z.min()), float(marker_z.max())
    z_mid = (z_min + z_max) / 2.0
    lower = marker_z[marker_z < z_mid]
    upper = marker_z[marker_z > z_mid]
    if lower.size == 0 or upper.size == 0:
        return None
    surf_lower = float(lower.max())
    surf_upper = float(upper.min())
    gap = surf_upper - surf_lower
    if gap < 0 or gap > 100.0:
        return None
    interface = [(surf_lower - eps, surf_lower + eps),
                 (surf_upper - eps, surf_upper + eps)]
    bulk = (surf_lower + eps + margin, surf_upper - eps - margin)
    if bulk[1] <= bulk[0]:
        return None
    return interface, bulk
