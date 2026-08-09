"""detect_zones tests (ported from MD-Analyzer _detect_zones, MIT)."""

from __future__ import annotations

import argparse

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from region import add_region_args, detect_zones, select_frame  # noqa: E402


def test_detect_zones_two_surfaces():
    """Marker atoms split into lower (z~5) and upper (z~25) surfaces."""
    rng = np.random.default_rng(1)
    lower = rng.normal(5.0, 0.1, 200)
    upper = rng.normal(25.0, 0.1, 200)
    z = np.concatenate([lower, upper])
    zones = detect_zones(z, eps=1.0, margin=3.0)
    assert zones is not None
    interface, bulk = zones
    # lower surface ~5, upper ~25
    assert abs(interface[0][0] - 4.0) < 0.5 and abs(interface[0][1] - 6.0) < 0.5
    assert abs(interface[1][0] - 24.0) < 0.5 and abs(interface[1][1] - 26.0) < 0.5
    # bulk between surfaces minus eps+margin
    assert bulk[0] == pytest.approx(interface[0][1] + 3.0)
    assert bulk[1] == pytest.approx(interface[1][0] - 3.0)


def test_detect_zones_single_surface_none():
    z = np.random.default_rng(2).normal(10.0, 0.5, 100)  # one surface only
    assert detect_zones(z) is None


def test_detect_zones_gap_too_large_none():
    z = np.concatenate([
        np.random.default_rng(3).normal(0.0, 0.1, 100),
        np.random.default_rng(4).normal(200.0, 0.1, 100),
    ])
    assert detect_zones(z) is None


def _scene():
    """Marker atoms (type 1, z~5 and z~25) + waters (type 2)."""
    rng = np.random.default_rng(5)
    lower = rng.normal(5.0, 0.1, 200)
    upper = rng.normal(25.0, 0.1, 200)
    # markers are Gaussian (surf ~5.0/25.0 +- 0.35), so bulk band is
    # ~(9.35, 20.65) in the worst case; keep bulk candidates well inside it
    water_z = np.array([5.5, 11.0, 15.0, 19.0, 25.5])  # interface, bulk, bulk, bulk, interface
    n_marker = lower.size + upper.size
    coords = np.column_stack([
        np.zeros(n_marker + 5),
        np.zeros(n_marker + 5),
        np.concatenate([lower, upper, water_z]),
    ])
    types = np.concatenate([np.ones(n_marker, dtype=int), np.full(5, 2, dtype=int)])
    return coords, types


def test_layer_auto_select_interface():
    ap = argparse.ArgumentParser()
    add_region_args(ap)
    args = ap.parse_args([
        "--region", "layer", "--layer-auto", "--marker-type", "1",
        "--zone-eps", "1.0", "--zone-margin", "3.0", "--zone", "interface"])
    coords, types = _scene()
    box = np.array([[0, 30], [0, 30], [0, 30]])
    mask = select_frame(coords, box, args, types)
    water_mask = mask[-5:]  # only the water atoms' selection matters
    # interface = [4,6] and [24,26] -> z=5.5 and z=25.5 selected
    assert water_mask.tolist() == [True, False, False, False, True]


def test_layer_auto_select_bulk():
    ap = argparse.ArgumentParser()
    add_region_args(ap)
    args = ap.parse_args([
        "--region", "layer", "--layer-auto", "--marker-type", "1",
        "--zone-eps", "1.0", "--zone-margin", "3.0", "--zone", "bulk"])
    coords, types = _scene()
    box = np.array([[0, 30], [0, 30], [0, 30]])
    mask = select_frame(coords, box, args, types)
    water_mask = mask[-5:]
    # bulk band ~ [9.3, 20.7] (surf +- Gaussian spread) -> z=11, 15, 19 selected
    assert water_mask.tolist() == [False, True, True, True, False]


def test_layer_auto_requires_marker_type():
    ap = argparse.ArgumentParser()
    add_region_args(ap)
    args = ap.parse_args(["--region", "layer", "--layer-auto"])
    coords = np.zeros((3, 3))
    box = np.array([[0, 10], [0, 10], [0, 10]])
    with pytest.raises(ValueError, match="marker-type"):
        select_frame(coords, box, args, np.array([1, 1, 1]))
