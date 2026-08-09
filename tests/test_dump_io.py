"""Parser tests: column-driven parsing, non-contiguous ids, dt inference,
triclinic rejection, missing-column errors, elements map."""

from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from dump_io import parse_dump, read_types_elements  # noqa: E402


def test_parse_mini_basics(mini_dump_path):
    d = parse_dump(mini_dump_path, timestep_fs=0.5)
    assert d.natoms == 4
    assert len(d.timesteps) == 5
    assert d.timesteps.tolist() == [0, 100, 200, 300, 400]
    # median dT = 100 -> dt_fs = 100 * 0.5
    assert d.dt_fs == pytest.approx(50.0)
    assert d.wrapped is True
    assert d.box.shape == (5, 3, 2)
    assert d.cell[0, 0, 0] == pytest.approx(20.0)


def test_noncontiguous_ids_preserved(mini_dump_path):
    d = parse_dump(mini_dump_path)
    assert d.atom_ids[0].tolist() == [1, 2, 3, 7]
    assert d.types[0].tolist() == [1, 1, 2, 2]


def test_coords_column_driven(mini_dump_path):
    d = parse_dump(mini_dump_path)
    # frame 0 atom 1 at (1,2,3)
    assert d.coords[0, 0].tolist() == [1.0, 2.0, 3.0]
    # column order scrambled on purpose would still parse; here standard order
    assert d.columns == ["id", "type", "x", "y", "z"]


def test_start_stop_stride(mini_dump_path):
    d = parse_dump(mini_dump_path, start=1, stop=5, stride=2)
    assert len(d.timesteps) == 2
    assert d.timesteps.tolist() == [100, 300]
    d2 = parse_dump(mini_dump_path, stride=2)
    assert d2.timesteps.tolist() == [0, 200, 400]


def test_wrapped_detection_unwrapped_columns(tmp_path):
    # dump with xu yu zu (already unwrapped) must set wrapped=False
    content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type xu yu zu
1 1 0.5 0.5 0.5
2 2 9.5 9.5 9.5
"""
    p = tmp_path / "unwrapped.dump"
    p.write_text(content)
    d = parse_dump(p)
    assert d.wrapped is False
    assert d.coords[0, 1].tolist() == [9.5, 9.5, 9.5]


def test_triclinic_rejected(tmp_path):
    content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS xy xz yz pp pp pp
0 10 1.0
0 10 0.5
0 10 0.0
0.1 0.2 0.3
ITEM: ATOMS id type x y z
1 1 0.5 0.5 0.5
2 2 9.5 9.5 9.5
"""
    p = tmp_path / "triclinic.dump"
    p.write_text(content)
    with pytest.raises(ValueError, match="triclinic"):
        parse_dump(p)


def test_missing_id_column_rejected(tmp_path):
    content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS type x y z
1 0.5 0.5 0.5
"""
    p = tmp_path / "noid.dump"
    p.write_text(content)
    with pytest.raises(ValueError, match="id"):
        parse_dump(p)


def test_missing_coords_rejected(tmp_path):
    content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type vx vy vz
1 1 0.1 0.2 0.3
"""
    p = tmp_path / "nocoords.dump"
    p.write_text(content)
    with pytest.raises(ValueError, match="coordinate"):
        parse_dump(p)


def test_elements_map_cli():
    m = read_types_elements(elements_map="1:H 2:O")
    assert m == {1: "H", 2: "O"}


def test_elements_map_from_data(mini_data_path):
    m = read_types_elements(data_path=str(mini_data_path))
    assert m == {1: "H", 2: "O"}


def test_elements_map_priority(mini_data_path):
    m = read_types_elements(data_path=str(mini_data_path), elements_map="1:Na 2:Cl")
    assert m == {1: "Na", 2: "Cl"}
