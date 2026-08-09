"""Region selection tests: box / sphere / layer / per-frame semantics."""

from __future__ import annotations

import argparse

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from region import add_region_args, select_frame, select_traj  # noqa: E402


def make_parser(**defaults):
    ap = argparse.ArgumentParser()
    add_region_args(ap)
    # region defaults come through argparse defaults already
    return ap


def parse_args(ap, argv):
    return ap.parse_args(argv)


def test_full_selects_all():
    ap = make_parser()
    args = ap.parse_args([])
    coords = np.random.default_rng(0).uniform(0, 10, size=(50, 3))
    box = np.array([[0, 10], [0, 10], [0, 10]])
    assert select_frame(coords, box, args).all()


def test_box_region():
    ap = make_parser()
    args = ap.parse_args(
        ["--region", "box", "--box-min", "2 2 2", "--box-max", "4 4 4"])
    coords = np.array([[3, 3, 3], [5, 3, 3], [3, 1, 3], [3, 3, 5]])
    box = np.array([[0, 10], [0, 10], [0, 10]])
    mask = select_frame(coords, box, args)
    assert mask.tolist() == [True, False, False, False]


def test_sphere_region():
    ap = make_parser()
    args = ap.parse_args(
        ["--region", "sphere", "--center", "5 5 5", "--r-in", "1", "--r-out", "3"])
    coords = np.array([
        [5, 5, 5],     # dist 0 -> inside shell (r_in=1 rejects)
        [5, 5, 7],     # dist 2 -> inside
        [5, 5, 10],    # dist 5 -> outside
    ])
    box = np.array([[0, 10], [0, 10], [0, 10]])
    mask = select_frame(coords, box, args)
    assert mask.tolist() == [False, True, False]


def test_layer_top_bottom():
    ap = make_parser()
    args = ap.parse_args(["--region", "layer", "--layer", "top", "--trunc", "2"])
    coords = np.array([[0, 0, 9.5], [0, 0, 7.0]])
    box = np.array([[0, 10], [0, 10], [0, 10]])
    mask = select_frame(coords, box, args)
    assert mask.tolist() == [True, False]

    args = ap.parse_args(["--region", "layer", "--layer", "bottom", "--trunc", "1"])
    coords = np.array([[0, 0, 0.5], [0, 0, 2.0]])
    mask = select_frame(coords, box, args)
    assert mask.tolist() == [True, False]


def test_layer_zband():
    ap = make_parser()
    args = ap.parse_args(["--region", "layer", "--zmin", "3", "--zmax", "5"])
    coords = np.array([[0, 0, 3.5], [0, 0, 5.5]])
    box = np.array([[0, 10], [0, 10], [0, 10]])
    mask = select_frame(coords, box, args)
    assert mask.tolist() == [True, False]


def test_box_requires_bounds():
    ap = make_parser()
    args = ap.parse_args(["--region", "box"])
    coords = np.zeros((3, 3))
    box = np.array([[0, 10], [0, 10], [0, 10]])
    with pytest.raises(ValueError, match="box-min"):
        select_frame(coords, box, args)


def test_static_mask_broadcast_over_frames():
    ap = make_parser()
    args = ap.parse_args(["--region", "box", "--box-min", "2 2 2", "--box-max", "4 4 4"])
    coords = np.zeros((5, 3, 3))
    box = np.zeros((5, 3, 2))
    mask = select_traj(coords, box, args)
    assert mask.shape == (5, 3)
    # first-frame mask broadcast: all frames identical
    assert np.array_equal(mask[0], mask[4])


def test_per_frame_reselection():
    ap = make_parser()
    args = ap.parse_args(
        ["--region", "box", "--box-min", "0 0 0", "--box-max", "1 1 1", "--per-frame"])
    coords = np.array([
        [[0.5, 0.5, 0.5], [5.0, 5.0, 5.0]],
        [[5.0, 5.0, 5.0], [0.5, 0.5, 0.5]],
    ])
    box = np.zeros((2, 3, 2))
    mask = select_traj(coords, box, args)
    assert mask.shape == (2, 2)
    assert mask[0].tolist() == [True, False]
    assert mask[1].tolist() == [False, True]
