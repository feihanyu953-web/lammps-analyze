#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Water-molecule identification for lammps-analyze (pure numpy).

Pairs each O atom with its two nearest H atoms within a covalent cutoff
(default 1.3 A; the O-H bond is ~0.96 A), greedily per frame. Requires the
type->element map so O and H type ids are known (--data or --elements-map).

Pass ``box`` (the per-frame lo/hi cell, e.g. Dump.box) to use minimum-image
distances: this is robust to wrapped input AND to inconsistent unwrap
(image-flag drift in xu/yu/zu dumps, where a molecule's O and H can appear
box-lengths apart). Without ``box``, raw distances are used (assumes a
consistent unwrap). O's that cannot be paired (e.g. an OH- O with a single
covalent H in electrolyte systems) are skipped per frame, not fatal.

Used by l3_tcf (residence / dipole relaxation) and l4_interface (orientation).
"""

from __future__ import annotations

import numpy as np


def _mic(dv: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Minimum-image a displacement array: dv -= L * rint(dv / L).

    dv: (..., 3); L: (3,) box lengths. Image-independent: works for wrapped,
    consistently unwrapped, and image-drifted coordinates alike.
    """
    return dv - L[None, None, :] * np.rint(dv / L[None, None, :])


def find_water_molecules(types: np.ndarray, coords: np.ndarray,
                         o_type: int, h_types: list[int],
                         oh_cutoff: float = 1.3,
                         box: np.ndarray | None = None,
                         return_frames: bool = False
                         ) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None:
    """Pair each O with its two nearest H atoms.

    types: (nframe, natom) type ids; coords: (nframe, natom, 3) unwrapped.
    box: optional (nframe, 3, 2) lo/hi orthorhombic cell (Dump.box); when
    given, O-H distances are minimum-image (PBC-aware), robust to wrapped or
    inconsistently-unwrapped coordinates.

    Returns mols (n_keep, n_mol, 3) int array of atom indices [o, h1, h2]
    sorted by O index, or None if no O / no frame could be paired. Only
    frames whose pairable O count equals the most frequent count are kept
    (a frame can lose O's to an OH- O, a proton-transfer snapshot, or an
    image-drifted molecule), so the stack is regular. With return_frames=True
    returns (mols, frames) where frames is (n_keep,) the trajectory indices
    each mols row belongs to -- required when coords/box are indexed by
    trajectory frame (l3_tcf, l4_interface). return_frames=False (default)
    keeps the pre-refactor single-array return for backward compatibility.
    """
    nframe = types.shape[0]
    h_set = set(h_types)
    per_frame: list[tuple[int, np.ndarray]] = []  # (frame_idx, (n_mol, 3))
    for f in range(nframe):
        o_idx = np.flatnonzero(types[f] == o_type)
        h_idx = np.flatnonzero(np.isin(types[f], list(h_set)))
        if o_idx.size == 0:
            continue
        if h_idx.size < 2 * o_idx.size:
            continue  # not enough H for one-to-two pairing this frame
        pos_o = coords[f, o_idx]
        pos_h = coords[f, h_idx]
        dv = pos_o[:, None, :] - pos_h[None, :, :]  # (n_o, n_h, 3)
        if box is not None:
            dv = _mic(dv, box[f][:, 1] - box[f][:, 0])
        d = np.linalg.norm(dv, axis=2)
        frame_mols = []
        assigned_h = np.zeros(h_idx.size, dtype=bool)
        for i, oi in enumerate(o_idx):
            order = np.argsort(d[i])
            picks = []
            for j in order:
                if len(picks) == 2:
                    break
                if not assigned_h[j] and d[i, j] < oh_cutoff:
                    assigned_h[j] = True
                    picks.append(j)
            if len(picks) != 2:
                continue  # skip un-pairable O (e.g. OH- O with a single
                # covalent H in electrolyte systems, or an image-drifted one)
            frame_mols.append((oi, h_idx[picks[0]], h_idx[picks[1]]))
        if frame_mols:
            per_frame.append((f, np.array(frame_mols, dtype=np.int64)))
    if not per_frame:
        return None
    # keep the most frequent molecule count so np.stack stays regular;
    # outlier frames (proton-transfer / drift snapshots) are dropped
    counts = np.bincount([m.shape[0] for _, m in per_frame])
    best_n = int(np.argmax(counts))
    sel = [(f, m) for f, m in per_frame if m.shape[0] == best_n]
    frames = np.array([f for f, _ in sel], dtype=np.int64)
    mols = np.stack([m for _, m in sel])
    if return_frames:
        return mols, frames
    return mols


def water_dipoles(mols: np.ndarray, coords: np.ndarray,
                  box: np.ndarray | None = None,
                  frames: np.ndarray | None = None) -> np.ndarray:
    """Dipole direction vectors (unnormalized) for each molecule per frame.

    mu = (H1 - O) + (H2 - O); returns (nframe, n_mol, 3). Normalize with
    axis=-1 when only the direction is needed. box: optional (nframe, 3, 2)
    lo/hi cell; when given the O->H vectors are minimum-image, so dipoles are
    correct even when the dump's unwrap is inconsistent (image-flag drift).
    frames: optional (nframe,) trajectory indices for mols rows (returned by
    find_water_molecules); defaults to consecutive 0..nframe-1.
    """
    if frames is None:
        frames = np.arange(mols.shape[0])
    o = coords[frames[:, None], mols[:, :, 0]]
    h1 = coords[frames[:, None], mols[:, :, 1]]
    h2 = coords[frames[:, None], mols[:, :, 2]]
    d1 = h1 - o
    d2 = h2 - o
    if box is not None:
        # Dump.box is (nframe, 3, 2) lo/hi per axis; fancy-index frames
        # first, then take lo/hi on the last axis. Per-frame L broadcasts
        # against (nframe, n_mol, 3) along axis 1 (unlike _mic's (3,) case).
        L = box[frames][:, :, 1] - box[frames][:, :, 0]  # (nframe, 3)
        d1 = d1 - L[:, None, :] * np.rint(d1 / L[:, None, :])
        d2 = d2 - L[:, None, :] * np.rint(d2 / L[:, None, :])
    return d1 + d2


def water_o_positions(mols: np.ndarray, coords: np.ndarray,
                      frames: np.ndarray | None = None) -> np.ndarray:
    """O positions per molecule: (nframe, n_mol, 3).

    frames: optional (nframe,) trajectory indices for mols rows (returned by
    find_water_molecules); defaults to consecutive 0..nframe-1.
    """
    if frames is None:
        frames = np.arange(mols.shape[0])
    return coords[frames[:, None], mols[:, :, 0]]
