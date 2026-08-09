#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Water-molecule identification for lammps-analyze (pure numpy).

Pairs each O atom with its two nearest H atoms within a covalent cutoff
(default 1.3 A; the O-H bond is ~0.96 A), greedily per frame. Requires the
type->element map so O and H type ids are known (--data or --elements-map).

Used by l3_tcf (residence / dipole relaxation) and l4_interface (orientation).
"""

from __future__ import annotations

import numpy as np


def find_water_molecules(types: np.ndarray, coords: np.ndarray,
                         o_type: int, h_types: list[int],
                         oh_cutoff: float = 1.3) -> np.ndarray | None:
    """Pair each O with its two nearest H atoms.

    types: (nframe, natom) type ids; coords: (nframe, natom, 3) unwrapped.
    Returns (nframe, n_mol, 3) int array of atom indices [o, h1, h2] sorted
    by O index, or None if no O or fewer than 2 H per O are found.
    """
    nframe, natom = types.shape
    h_set = set(h_types)
    mols: list[np.ndarray] = []  # per frame: (n_mol, 3)
    for f in range(nframe):
        o_idx = np.flatnonzero(types[f] == o_type)
        h_idx = np.flatnonzero(np.isin(types[f], list(h_set)))
        if o_idx.size == 0:
            return None
        if h_idx.size < 2 * o_idx.size:
            return None  # not enough H for one-to-two pairing
        pos_o = coords[f, o_idx]
        pos_h = coords[f, h_idx]
        # distance matrix (n_o, n_h); no PBC: intra-molecule distances are tiny
        d = np.linalg.norm(pos_o[:, None, :] - pos_h[None, :, :], axis=2)
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
                return None  # could not pair this O
            frame_mols.append((oi, h_idx[picks[0]], h_idx[picks[1]]))
        mols.append(np.array(frame_mols, dtype=np.int64))
    return np.stack(mols)


def water_dipoles(mols: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Dipole direction vectors (unnormalized) for each molecule per frame.

    mu = (H1 - O) + (H2 - O); returns (nframe, n_mol, 3). Normalize with
    axis=-1 when only the direction is needed.
    """
    o = coords[np.arange(mols.shape[0])[:, None], mols[:, :, 0]]
    h1 = coords[np.arange(mols.shape[0])[:, None], mols[:, :, 1]]
    h2 = coords[np.arange(mols.shape[0])[:, None], mols[:, :, 2]]
    return (h1 - o) + (h2 - o)


def water_o_positions(mols: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """O positions per molecule: (nframe, n_mol, 3)."""
    return coords[np.arange(mols.shape[0])[:, None], mols[:, :, 0]]
