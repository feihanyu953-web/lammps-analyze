"""molecules.py unit tests: O-H pairing and dipole directions."""

from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from molecules import find_water_molecules, water_dipoles  # noqa: E402


def _two_water_frame():
    """Two waters with covalent O-H distances (~0.96 A) in both molecules."""
    types = np.array([[2, 1, 1, 2, 1, 1]])  # O H H O H H
    coords = np.array([[
        [0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0],
        [3.2, 0.0, 0.0], [4.16, 0.0, 0.0], [2.96, 0.93, 0.0],
    ]])
    return types, coords


def test_pairing_finds_two_waters():
    types, coords = _two_water_frame()
    paired = find_water_molecules(types, coords, o_type=2, h_types=[1],
                                  return_frames=True)
    assert paired is not None
    mols, frames = paired
    assert mols.shape == (1, 2, 3)
    assert frames.tolist() == [0]
    # O indices sorted: molecule 0 = O idx 0, molecule 1 = O idx 3
    assert mols[0, 0, 0] == 0
    assert mols[0, 1, 0] == 3
    # H indices: O0 pairs with H1,H2; O3 with H4,H5
    assert sorted(mols[0, 0, 1:].tolist()) == [1, 2]
    assert sorted(mols[0, 1, 1:].tolist()) == [4, 5]


def test_pairing_legacy_single_array_return():
    """Backward compatibility: without return_frames the function returns the
    bare (nframe, n_mol, 3) array, as before the frames-refactor."""
    types, coords = _two_water_frame()
    mols = find_water_molecules(types, coords, o_type=2, h_types=[1])
    assert mols is not None
    assert mols.shape == (1, 2, 3)
    assert mols[0, 0, 0] == 0


def test_pairing_fails_without_h():
    types, coords = _two_water_frame()
    types2 = types.copy()
    types2[0, 1] = 2  # replace an H with O -> not enough H
    assert find_water_molecules(types2, coords, o_type=2, h_types=[1]) is None


def test_pairing_fails_no_o():
    types, coords = _two_water_frame()
    types2 = np.full_like(types, 1)
    assert find_water_molecules(types2, coords, o_type=2, h_types=[1]) is None


def test_dipole_direction():
    types, coords = _two_water_frame()
    paired = find_water_molecules(types, coords, o_type=2, h_types=[1],
                                  return_frames=True)
    assert paired is not None
    mols, frames = paired
    mu = water_dipoles(mols, coords, frames=frames)
    # molecule 0: (H1-O)+(H2-O) = (0.96,0,0)+(-0.24,0.93,0) = (0.72, 0.93, 0)
    assert np.allclose(mu[0, 0], [0.72, 0.93, 0.0])
    # molecule 1: (H4-O2)+(H5-O2) = (0.96,0,0)+(-0.24,0.93,0) = same vector
    assert np.allclose(mu[0, 0], mu[0, 1])
    assert np.linalg.norm(mu[0, 0]) > 0
