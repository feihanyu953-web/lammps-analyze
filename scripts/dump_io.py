#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared LAMMPS dump I/O for lammps-analyze (pure numpy, no third-party deps).

Parsing layer of the skill: reads dump.lammpstrj into a Dump dataclass,
PBC-unwraps orthorhombic trajectories, and writes native lammpstrj files
consumable by OVITO / MDAnalysis / downstream tools.

PBC unwrap algorithm (from the project's calc_msd.py, vectorized):
    frac -> diff -> d -= round(d) -> cumsum -> first-frame absolute + offsets
Unlike a >L/2 threshold method, round(dfrac) handles arbitrary displacements,
uses per-frame cell vectors, and is a no-op on already-unwrapped input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ELEMENT_RE = re.compile(r"#\s*([A-Z][a-z]?)")


@dataclass
class Dump:
    """One LAMMPS dump trajectory (orthorhombic cells only, MVP scope)."""

    timesteps: np.ndarray      # (nframe,) int64
    dt_fs: float | None        # median(diff(timesteps)) * timestep_fs
    natoms: int
    box: np.ndarray            # (nframe, 3, 2) lo/hi
    cell: np.ndarray           # (nframe, 3, 3) diagonal (orthorhombic)
    atom_ids: np.ndarray       # (nframe, natom) int64 -- stored, NOT assumed contiguous
    types: np.ndarray          # (nframe, natom) int
    coords: np.ndarray         # (nframe, natom, 3) float64
    velocities: np.ndarray | None  # (nframe, natom, 3) or None if no vx/vy/vz columns
    columns: list[str]         # ITEM: ATOMS column names
    wrapped: bool              # True if x/y/z (PBC-wrapped), False if xu/yu/zu

    @property
    def timesteps_rel(self) -> np.ndarray:
        """Timesteps anchored at the first frame (t[0] = 0).

        Raw `timesteps` are cumulative LAMMPS step counters: a dump written only
        during production still carries every pre-production step (minimize +
        equilibration), so `timesteps * timestep_fs` is time-since-simulation-start,
        NOT time-since-production-start (observed as a 200-2200 ps axis on a
        2 ns production). Anchor x axes on this property.
        """
        return self.timesteps - self.timesteps[0]


class _Pushback:
    """Tiny pushback iterator so a peeked line can be returned to the stream."""

    def __init__(self, it):
        self._it = iter(it)
        self._buf = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._buf is not None:
            v = self._buf
            self._buf = None
            return v
        return next(self._it)

    def push(self, v):
        self._buf = v


def parse_dump(path, start: int = 0, stop: int = -1, stride: int = 1,
               timestep_fs: float = 0.5) -> Dump:
    """Parse a LAMMPS dump file into a Dump.

    Column-name driven: coordinates are located by name (x/y/z or xu/yu/zu),
    ids are stored as data (they may be non-contiguous). start/stop/stride are
    applied while reading (streaming). Triclinic cells (4-row BOX BOUNDS)
    raise ValueError: not supported in this MVP.

    Note: `timesteps` are cumulative LAMMPS step counters (they include any
    pre-dump minimize/equilibration stages); use `Dump.timesteps_rel` for
    production-relative time axes.
    """
    path = Path(path)
    frames: list[dict] = []
    n_expected = None

    with path.open("r", encoding="utf-8") as fh:
        line_iter = _Pushback(fh)
        for raw in line_iter:
            line = raw.rstrip("\n").strip()
            if not line:
                continue
            if line.startswith("ITEM:"):
                payload = line[5:].strip()
                if payload.startswith("TIMESTEP"):
                    step = int(next(line_iter).strip())
                    frames.append({"timestep": step})
                elif payload.startswith("NUMBER OF ATOMS"):
                    n = int(next(line_iter).strip())
                    if n_expected is None:
                        n_expected = n
                    elif n != n_expected:
                        raise ValueError(
                            f"atom count changed across frames: {n_expected} -> {n}")
                    frames[-1]["natoms"] = n
                elif payload.startswith("BOX BOUNDS"):
                    bounds = []
                    for _ in range(3):
                        lo, hi = next(line_iter).split()[:2]
                        bounds.append((float(lo), float(hi)))
                    # orthorhombic has 3 numeric rows; triclinic adds a 4th tilt row.
                    # Peek the next line: a numeric 4th row => triclinic (reject);
                    # an ITEM: line => push it back and continue.
                    extra = next(line_iter).strip()
                    if extra and not extra.startswith("ITEM:"):
                        raise ValueError(
                            "triclinic dump (4-row BOX BOUNDS) not supported in this MVP; "
                            "convert to orthorhombic or unwrap before analysis")
                    if extra:
                        line_iter.push(extra)
                    frames[-1]["box"] = np.array(bounds, dtype=float)
                elif payload.startswith("ATOMS"):
                    cols = payload[5:].strip().split()
                    atoms = []
                    for _ in range(n_expected):
                        atoms.append([float(t) for t in next(line_iter).split()])
                    frames[-1]["cols"] = cols
                    frames[-1]["atoms"] = np.array(atoms)
                else:
                    raise ValueError(f"unknown ITEM: {payload}")

    if not frames:
        raise ValueError(f"no frames parsed from {path}")
    if any("atoms" not in fr for fr in frames):
        raise ValueError(f"{path}: at least one frame lacks ITEM: ATOMS")

    n_total = len(frames)
    idx = list(range(n_total))[start: (stop if stop >= 0 else None): stride]
    if not idx:
        raise ValueError(f"empty frame selection: start={start} stop={stop} stride={stride}")

    nframe = len(idx)
    timesteps = np.array([frames[i]["timestep"] for i in idx], dtype=np.int64)
    natoms = n_expected
    box = np.stack([frames[i]["box"] for i in idx])            # (F,3,2)
    cell = np.zeros((nframe, 3, 3), dtype=float)
    cell[:, 0, 0] = box[:, 0, 1] - box[:, 0, 0]
    cell[:, 1, 1] = box[:, 1, 1] - box[:, 1, 0]
    cell[:, 2, 2] = box[:, 2, 1] - box[:, 2, 0]

    cols = frames[idx[0]]["cols"]
    col_idx = {c: i for i, c in enumerate(cols)}
    if "id" not in col_idx:
        raise ValueError(f"ITEM: ATOMS lacks 'id' column (got: {cols})")
    if "type" not in col_idx:
        raise ValueError(f"ITEM: ATOMS lacks 'type' column (got: {cols})")

    xyz_cols = [c for c in ("x", "y", "z") if c in col_idx]
    uvw_cols = [c for c in ("xu", "yu", "zu") if c in col_idx]
    if len(xyz_cols) == 3:
        coord_cols = xyz_cols
        wrapped = True
    elif len(uvw_cols) == 3:
        coord_cols = uvw_cols
        wrapped = False
    else:
        raise ValueError(
            f"cannot locate coordinates: need x/y/z or xu/yu/zu (got: {cols})")

    coords = np.empty((nframe, natoms, 3), dtype=float)
    atom_ids = np.empty((nframe, natoms), dtype=np.int64)
    types = np.empty((nframe, natoms), dtype=int)
    vel_cols = [c for c in ("vx", "vy", "vz") if c in col_idx]
    velocities = None
    if len(vel_cols) == 3:
        velocities = np.empty((nframe, natoms, 3), dtype=float)
    for k, i in enumerate(idx):
        a = frames[i]["atoms"]
        atom_ids[k] = a[:, col_idx["id"]].astype(np.int64)
        types[k] = a[:, col_idx["type"]].astype(int)
        coords[k] = a[:, [col_idx[c] for c in coord_cols]]
        if velocities is not None:
            velocities[k] = a[:, [col_idx[c] for c in vel_cols]]

    diffs = np.diff(timesteps)
    dt_fs = None
    nz = diffs[diffs != 0]
    if len(nz):
        dt_fs = float(np.median(nz)) * timestep_fs

    return Dump(timesteps=timesteps, dt_fs=dt_fs, natoms=natoms, box=box,
                cell=cell, atom_ids=atom_ids, types=types, coords=coords,
                velocities=velocities, columns=cols, wrapped=wrapped)


def unwrap_ortho(coords: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """PBC-unwrap wrapped cartesian coordinates for orthorhombic cells.

    coords: (nframe, natom, 3) wrapped; cell: (nframe, 3, 3) diagonal.
    Returns unwrapped cartesian coordinates: first frame absolute, subsequent
    frames offset by the accumulated nearest-image displacement.
    """
    inv = np.linalg.inv(cell)                                   # (F,3,3)
    frac = np.einsum("fad,fdb->fab", coords, inv)
    d = np.diff(frac, axis=0)
    d -= np.round(d)
    uw = np.concatenate([frac[:1], frac[:1] + np.cumsum(d, axis=0)], axis=0)
    return np.einsum("fad,fdb->fab", uw, cell)


def unwrap_com(coords: np.ndarray, com_ref_types: np.ndarray,
               types: np.ndarray) -> np.ndarray:
    """Remove per-frame center-of-mass drift of reference types from coords.

    coords: (nframe, natom, 3); com_ref_types: set of type ids to compute the
    COM over; types: (nframe, natom). Returns coords shifted so the reference
    COM sits at the origin every frame. Requires unwrapped coords (call after
    unwrap_ortho).
    """
    mask = np.isin(types, list(com_ref_types))
    out = coords.copy()
    if not mask.any():
        raise ValueError(f"COM reference types {sorted(com_ref_types)} not present in trajectory")
    for f in range(coords.shape[0]):
        out[f] -= coords[f][mask[f]].mean(axis=0)
    return out


def write_unwrapped_dump(path, timesteps, box, atom_ids, types, coords_uw):
    """Write native lammpstrj with 'id type xu yu zu' columns (OVITO-readable)."""
    path = Path(path)
    nframe, natoms, _ = coords_uw.shape
    with path.open("w", encoding="utf-8") as fh:
        for f in range(nframe):
            fh.write(f"ITEM: TIMESTEP\n{timesteps[f]}\n")
            fh.write(f"ITEM: NUMBER OF ATOMS\n{natoms}\n")
            fh.write("ITEM: BOX BOUNDS pp pp pp\n")
            for d in range(3):
                fh.write(f"{box[f, d, 0]:.10f} {box[f, d, 1]:.10f}\n")
            fh.write("ITEM: ATOMS id type xu yu zu\n")
            for a in range(natoms):
                x, y, z = coords_uw[f, a]
                fh.write(f"{atom_ids[f, a]} {types[f, a]} {x:.6f} {y:.6f} {z:.6f}\n")


def read_types_elements(data_path=None, elements_map=None) -> dict[int, str]:
    """Return {type_id: element_symbol}.

    Priority: elements_map (CLI, e.g. "1:H 2:O") > data.lammps Masses line
    '# element' comments > {} (unknown; analysis falls back to type ids).
    """
    result: dict[int, str] = {}
    if elements_map:
        for token in str(elements_map).split():
            if ":" not in token:
                raise ValueError(f"--elements-map token '{token}' must look like '1:H'")
            t, el = token.split(":")
            result[int(t)] = el
        return result
    if data_path:
        data_path = Path(data_path)
        if data_path.is_file():
            in_masses = False
            for line in data_path.open("r", encoding="utf-8", errors="replace"):
                s = line.strip()
                if s.startswith("Masses"):
                    in_masses = True
                    continue
                if in_masses:
                    if s.startswith(("Atoms", "Velocities", "Bonds", "Angles")):
                        break
                    m = _ELEMENT_RE.search(s)
                    parts = s.split()
                    if len(parts) >= 2 and parts[0].isdigit() and m:
                        result[int(parts[0])] = m.group(1)
    return result
