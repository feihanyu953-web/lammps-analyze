#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert VASP AIMD trajectories (vasprun.xml / XDATCAR) to LAMMPS lammpstrj.

The analysis modules of this skill consume LAMMPS dump files; this converter
bridges VASP AIMD output into that pipeline (RDF/CN/hbonds/MSD/orientation/
VACF all become available afterwards). Streaming throughout -- vasprun.xml
files from long AIMD runs are GB-scale and must not be loaded whole.

Sources:
  --vasprun  preferred source: per-step positions. Velocities are included
             when the file carries them (enabling VACF/VDOS downstream), but
             presence is VASP-version/settings dependent -- measured on a
             134 MB Andersen AIMD vasprun (2026-10-22): velocities only in the
             initial/final structure blocks, NOT per step, so the output is
             positions-only there. Units auto-calibrated against TEBEG from
             <parameters> when velocities exist (see below).
  --xdatcar  fallback: positions only (no velocities -- VACF/VDOS impossible).

Conventions / caveats (all loud, none silent):
  - The output has NO leading comment line: MDAnalysis's LAMMPS dump parser
    rejects files that start with comment lines (even the single comment line
    the LAMMPS format traditionally allows) -- found on the real-data
    end-to-end test. Provenance lives in <out>.elements_map + CONCLUSION.
  - vasprun.xml positions are CONTINUOUS (unwrapped) while XDATCAR positions
    are wrapped into the cell; both are valid inputs and the two agree to
    <1e-6 A under minimum image (cross-validated on the same run).
  - Output TIMESTEP is the frame index (0,1,2,...); pass the real dt via
    --timestep-fs (vasprun: POTIM; xdatcar: POTIM*NBLOCK). Downstream, anchor
    time axes on Dump.timesteps_rel.
  - Orthogonal cells only: the skill's minimum-image / unwrap / region math is
    orthogonal-only, and a triclinic cell would yield silently wrong hbonds and
    RDF. The converter fails loudly if any lattice has |off-diagonal| beyond
    tolerance (fail-closed, same philosophy as dump_io).
  - vasprun.xml velocities have no unit declaration in the file. The converter
    computes the kinetic temperature under both candidate interpretations
    (m/s and A/fs) and keeps the one closer to TEBEG -- report discloses the
    choice and the achieved T. Without TEBEG it falls back to m/s with a
    WARNING that the choice is unverified.
  - Positions come out wrapped (x y z); run l1_traj_prep --unwrap pbc (a no-op
    on already-unwrapped input) or modules with MIC engines as needed.
  - A companion <out>.elements_map file ("1:O 2:H ...") is written for the
    --elements-map flag of the analysis modules.

Examples:
  python aimd_to_dump.py --vasprun vasprun.xml --out traj.lammpstrj --timestep-fs 1.0
  python aimd_to_dump.py --xdatcar XDATCAR --out traj.xdat.lammpstrj --timestep-fs 2.0
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_utils import print_conclusion  # noqa: E402

AMU_TO_KG = 1.66053906660e-27
MS_PER_ANGFS = 1.0e5          # 1 A/fs = 1e5 m/s
J_PER_EV = 1.602176634e-19
KB_EV = 8.617333262e-5        # eV/K
METER_SCALE = 1.0e10          # vasprun.xml lengths are in meters -> A
ORTHOGONAL_RTOL = 1e-6


def _orthogonal_check(L: np.ndarray, where: str) -> np.ndarray:
    """Fail loudly on a triclinic lattice; return the 3 diagonal lengths."""
    diag = np.diag(L)
    off = np.abs(L - np.diagflat(diag)).max()
    if off > ORTHOGONAL_RTOL * max(np.abs(diag).max(), 1e-12):
        raise SystemExit(
            f"ERROR: {where}: triclinic lattice detected (max off-diagonal "
            f"{off:.3e} A). This skill's minimum-image/unwrap/region math is "
            "orthogonal-only and would produce silently wrong hbonds/RDF; "
            "convert/rotate the cell to orthorhombic first.")
    return diag


def _temp_K(v_angfs: np.ndarray, masses: np.ndarray) -> float:
    """Kinetic temperature [K] for velocities in A/fs (masses in amu)."""
    ke_eV = 0.5 * float(np.sum(masses[None, :] * (v_angfs ** 2).sum(-1))) \
        * (AMU_TO_KG * MS_PER_ANGFS ** 2 / J_PER_EV)
    n = masses.size
    dof = 3 * max(n - 1, 1)          # COM motion removed
    return 2.0 * ke_eV / (dof * KB_EV)


class _VasprunStreamer:
    """Streaming parser: atominfo/TEBEG once, then one calc block at a time."""

    def __init__(self, path: str):
        self.path = path
        self.elements: list[str] = []
        self.mass_of_elem: dict[str, float] = {}
        self.tebeg: float | None = None

    def _parse_atominfo(self, elem) -> None:
        for arr in elem.iter("array"):
            name = arr.get("name")
            fields = [f.text.strip() for f in arr.iter("field")]
            rows = [[c.text.strip() if c.text else ""
                     for c in rc.iter("c")] for rc in arr.iter("rc")]
            if not rows:
                continue
            if name == "atoms" and "element" in fields:
                col = fields.index("element")
                self.elements = [r[col] for r in rows]
            if name == "atomtypes" and "mass" in fields and "element" in fields:
                mi, ei = fields.index("mass"), fields.index("element")
                for r in rows:
                    try:
                        self.mass_of_elem[r[ei]] = float(r[mi])
                    except (ValueError, IndexError):
                        pass

    @staticmethod
    def _varray(parent, name: str) -> np.ndarray | None:
        node = parent.find(f"varray[@name='{name}']")
        if node is None:
            return None
        return np.array([[float(x) for x in v.text.split()] for v in node.iter("v")])

    def frames(self, stride: int = 1):
        """Yield (frac (N,3), L (3,3) in A, vel (N,3)|None) per kept step."""
        ctx = ET.iterparse(self.path, events=("start", "end"))
        n_written = 0
        for event, elem in ctx:
            if event == "start":
                if elem.tag == "modeling" and self._root is None:
                    self._root = elem
                continue
            if elem.tag == "atominfo":
                self._parse_atominfo(elem)
                elem.clear()
            elif elem.tag == "parameters":
                node = elem.find("i[@name='TEBEG']")
                if node is not None and node.text:
                    try:
                        self.tebeg = float(node.text.split()[0])
                    except ValueError:
                        pass
                elem.clear()
            elif elem.tag == "calculation":
                st = elem.find("structure")
                if st is None:
                    elem.clear()
                    continue
                frac = self._varray(st, "positions")
                basis = self._varray(st.find("crystal"), "basis")
                vel = self._varray(elem, "velocities")
                elem.clear()
                if self._root is not None:
                    self._root.clear()
                if frac is None or basis is None:
                    continue
                if n_written % stride == 0:
                    yield frac, basis, vel
                n_written += 1

    _root = None


def convert_vasprun(args) -> None:
    streamer = _VasprunStreamer(args.vasprun)
    out = Path(args.out)
    map_path = out.with_suffix(".elements_map")
    vel_unit_used: str | None = None
    T_report: float | None = None

    with open(out, "w", encoding="utf-8") as fh:
        n_at = None
        type_ids = None
        vel_mode: bool | None = None
        frame_idx = 0
        for frac, basis_m, vel in streamer.frames(stride=args.stride):
            if n_at is None:
                n_at = frac.shape[0]
                if not streamer.elements:
                    raise SystemExit("ERROR: vasprun.xml has no atominfo element "
                                     "list -- cannot map elements to types.")
                uniq = list(dict.fromkeys(streamer.elements))
                type_ids = [uniq.index(e) + 1 for e in streamer.elements]
            if frac.shape[0] != n_at:
                raise SystemExit(f"ERROR: atom count changed mid-run "
                                 f"({n_at} -> {frac.shape[0]})")
            L = basis_m
            if np.abs(L).max() < 1e-5:      # vasprun lengths are in meters
                L = L * METER_SCALE
            diag = _orthogonal_check(L, f"vasprun.xml frame {frame_idx}")
            cart = frac @ L

            vel_out = None
            if vel is not None and vel.shape == frac.shape:
                masses = np.array([streamer.mass_of_elem.get(e, 1.0)
                                   for e in streamer.elements])
                T_ms = _temp_K(vel * 1e-5, masses)     # if file were m/s
                T_ang = _temp_K(vel, masses)           # if file were A/fs
                if args.velocity_units == "ms":
                    unit, T_new = "ms", T_ms
                elif args.velocity_units == "angs_fs":
                    unit, T_new = "angs_fs", T_ang
                elif streamer.tebeg is not None:
                    unit = "ms" if abs(T_ms - streamer.tebeg) <= abs(
                        T_ang - streamer.tebeg) else "angs_fs"
                    T_new = T_ms if unit == "ms" else T_ang
                else:
                    unit, T_new = "ms", T_ms
                if frame_idx == 0 or unit != vel_unit_used:
                    if vel_unit_used is not None and unit != vel_unit_used:
                        raise SystemExit("ERROR: velocity unit interpretation "
                                         "flipped mid-trajectory -- file looks "
                                         "corrupted.")
                    vel_unit_used = unit
                    T_report = T_new
                # convert to A/fs (LAMMPS real convention): file-in-ms needs /1e5
                factor = 1.0 / MS_PER_ANGFS if unit == "ms" else 1.0
                vel_out = vel * factor
            if vel_mode is None:
                vel_mode = vel_out is not None
            elif (vel_out is not None) != vel_mode:
                raise SystemExit("ERROR: some calculation blocks have velocities "
                                 "and others do not -- dump columns would be "
                                 "inconsistent; aborting.")

            fh.write("ITEM: TIMESTEP\n"
                     f"{frame_idx}\n"
                     "ITEM: NUMBER OF ATOMS\n"
                     f"{n_at}\n"
                     "ITEM: BOX BOUNDS pp pp pp\n")
            for d in diag:
                fh.write(f"0.0 {d:.10f}\n")
            fh.write("ITEM: ATOMS id type x y z")
            has_vel = vel_mode
            if has_vel:
                fh.write(" vx vy vz")
            fh.write("\n")
            for i in range(n_at):
                line = (f"{i + 1} {type_ids[i]} "
                        f"{cart[i, 0]:.8f} {cart[i, 1]:.8f} {cart[i, 2]:.8f}")
                if has_vel:
                    line += (f" {vel_out[i, 0]:.8f} {vel_out[i, 1]:.8f}"
                             f" {vel_out[i, 2]:.8f}")
                fh.write(line + "\n")
            frame_idx += 1

    if frame_idx == 0:
        raise SystemExit(f"ERROR: no <calculation> structure frames found in "
                         f"{args.vasprun} (is this an MD run, IBRION=0?)")
    uniq = list(dict.fromkeys(streamer.elements))
    type_map = " ".join(f"{i + 1}:{e}" for i, e in enumerate(uniq))
    map_path.write_text(type_map + "\n", encoding="utf-8")
    print(f"==> wrote {out} ({frame_idx} frames, {n_at} atoms)")
    print(f"==> wrote {map_path} : {type_map}")
    units_txt = "n/a (no velocities)"
    if vel_unit_used:
        anchor = f"TEBEG={streamer.tebeg:g} K" if streamer.tebeg else "no TEBEG"
        warn = "" if streamer.tebeg else (
            " WARNING: no TEBEG in file -- interpretation UNVERIFIED, "
            "cross-check T before trusting VACF")
        units_txt = f"interpreted as '{vel_unit_used}' ({anchor}, T~{T_report:.1f} K){warn}"
    print_conclusion(
        f"vasprun -> {out.name}: {frame_idx} frames x {n_at} atoms, "
        f"types {type_map}; dt = {args.timestep_fs:g} fs/frame (user-supplied); "
        f"velocities: {units_txt}")


def convert_xdatcar(args) -> None:
    out = Path(args.out)
    map_path = out.with_suffix(".elements_map")
    with open(args.xdatcar, "r", encoding="utf-8") as fh:
        lines = (ln.strip() for ln in fh)
        title = next(lines)
        scale = float(next(lines))
        raw = np.array([[float(x) for x in next(lines).split()] for _ in range(3)])
        if scale < 0:                     # VASP4 style: negative = volume
            vol = abs(np.linalg.det(raw))
            f = (abs(scale) / vol) ** (1.0 / 3.0)
        else:
            f = scale
        L = raw * f
        diag = _orthogonal_check(L, f"{args.xdatcar} header")
        elems = next(lines).split()
        counts = [int(x) for x in next(lines).split()]
        n_at = sum(counts)
        elements = [e for e, c in zip(elems, counts) for _ in range(c)]
        uniq = list(dict.fromkeys(elements))
        type_ids = [uniq.index(e) + 1 for e in elements]

        frame_idx = 0
        kept = 0
        with open(out, "w", encoding="utf-8") as oh:
            for line in lines:
                if not line or not line.lower().startswith("direct"):
                    continue
                frac = np.array([[float(x) for x in
                                  next(lines).split()] for _ in range(n_at)])
                if frame_idx % args.stride == 0:
                    cart = frac @ L
                    oh.write("ITEM: TIMESTEP\n"
                             f"{kept}\n"
                             "ITEM: NUMBER OF ATOMS\n"
                             f"{n_at}\n"
                             "ITEM: BOX BOUNDS pp pp pp\n")
                    for d in diag:
                        oh.write(f"0.0 {d:.10f}\n")
                    oh.write("ITEM: ATOMS id type x y z\n")
                    for i in range(n_at):
                        oh.write(f"{i + 1} {type_ids[i]} {cart[i, 0]:.8f} "
                                 f"{cart[i, 1]:.8f} {cart[i, 2]:.8f}\n")
                    kept += 1
                frame_idx += 1
    if kept == 0:
        raise SystemExit(f"ERROR: no 'Direct configuration' frames parsed "
                         f"from {args.xdatcar}")
    type_map = " ".join(f"{i + 1}:{e}" for i, e in enumerate(uniq))
    map_path.write_text(type_map + "\n", encoding="utf-8")
    print(f"==> wrote {out} ({kept} frames, {n_at} atoms)")
    print(f"==> wrote {map_path} : {type_map}")
    print_conclusion(
        f"XDATCAR -> {out.name}: {kept} frames x {n_at} atoms, "
        f"types {type_map}; dt = {args.timestep_fs:g} fs/frame "
        f"(user-supplied POTIM*NBLOCK); velocities: none in XDATCAR "
        f"(VACF/VDOS impossible)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert VASP AIMD (vasprun.xml / XDATCAR) to lammpstrj")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--vasprun", help="vasprun.xml from an MD run (IBRION=0)")
    src.add_argument("--xdatcar", help="XDATCAR (positions only)")
    ap.add_argument("--out", default=None,
                    help="output lammpstrj (default: <input-stem>.lammpstrj)")
    ap.add_argument("--timestep-fs", type=float, default=1.0,
                    help="fs per written frame: POTIM (vasprun) or POTIM*NBLOCK "
                         "(xdatcar); feeds dump_io dt inference (default 1.0)")
    ap.add_argument("--velocity-units", choices=["auto", "ms", "angs_fs"],
                    default="auto",
                    help="vasprun velocity interpretation; 'auto' calibrates "
                         "against TEBEG (default auto)")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    args = ap.parse_args()
    if args.out is None:
        stem = Path(args.vasprun or args.xdatcar).stem
        args.out = str(Path(stem + ".lammpstrj"))
    if args.vasprun:
        convert_vasprun(args)
    else:
        convert_xdatcar(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
