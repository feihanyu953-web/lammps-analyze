"""l1_traj_prep integration tests: end-to-end on the mini fixture."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).parent.parent / "scripts"


def run_prep(tmp_path, *extra):
    cmd = [sys.executable, str(SCRIPTS / "l1_traj_prep.py"),
           "--dump", str(Path(__file__).parent / "fixtures" / "mini.dump"),
           "--data", str(Path(__file__).parent / "fixtures" / "mini.data"),
           "--outdir", str(tmp_path), "--name", "t", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"prep failed:\n{r.stdout}\n{r.stderr}"
    return r


def test_prep_pbc_unwrap_outputs(tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    from dump_io import parse_dump

    run_prep(tmp_path)
    uw_path = tmp_path / "t_uw.dump"
    meta_path = tmp_path / "t_meta.json"
    assert uw_path.is_file() and meta_path.is_file()

    # output re-parses as an unwrapped dump with standard columns
    d = parse_dump(uw_path)
    assert d.wrapped is False
    assert d.columns == ["id", "type", "xu", "yu", "zu"]
    assert d.natoms == 4
    assert len(d.timesteps) == 5

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["dt_fs"] == 50.0
    assert meta["natoms"] == 4
    assert meta["n_frames"] == 5
    assert meta["types_elements"] == {"1": "H", "2": "O"}
    assert meta["unwrap_mode"] == "pbc"


def test_prep_region_box_keeps_subset(tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    from dump_io import parse_dump

    run_prep(tmp_path, "--region", "box", "--box-min", "0 0 0", "--box-max", "5 6 7")
    # frame-0 atoms inside [0..5]x[0..6]x[0..7]: atom 1 (1,2,3) and atom 2 (4,5,6)
    d = parse_dump(tmp_path / "t_uw.dump")
    assert d.natoms == 2
    assert d.atom_ids[0].tolist() == [1, 2]
    meta = json.loads((tmp_path / "t_meta.json").read_text(encoding="utf-8"))
    assert meta["region"]["region"] == "box"


def test_prep_com_drift(tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    from dump_io import parse_dump

    r = run_prep(tmp_path, "--unwrap", "com", "--com-ref", "1")
    assert "COM drift removed" in r.stdout
    d = parse_dump(tmp_path / "t_uw.dump")
    # COM of type-1 atoms (ids 1,2) at every frame should sit at the origin
    t1 = np.where(d.types[0] == 1)[0]
    for f in range(5):
        com = d.coords[f, t1].mean(axis=0)
        assert np.allclose(com, 0.0, atol=1e-6), f"frame {f} COM={com}"


def test_prep_unwrapped_input_is_noop(tmp_path):
    """Already-unwrapped dump (xu yu zu) with --unwrap pbc passes through."""
    sys.path.insert(0, str(SCRIPTS))
    from dump_io import parse_dump

    src = tmp_path / "src.dump"
    src.write_text(
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
        "ITEM: ATOMS id type xu yu zu\n1 1 0.5 0.5 0.5\n2 2 9.5 9.5 9.5\n"
        "ITEM: TIMESTEP\n100\nITEM: NUMBER OF ATOMS\n2\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
        "ITEM: ATOMS id type xu yu zu\n1 1 0.7 0.6 0.5\n2 2 9.6 9.4 9.5\n",
        encoding="utf-8")
    cmd = [sys.executable, str(SCRIPTS / "l1_traj_prep.py"),
           "--dump", str(src), "--unwrap", "pbc", "--outdir", str(tmp_path),
           "--name", "u"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0
    assert "no-op" in r.stdout
    d = parse_dump(tmp_path / "u_uw.dump")
    assert np.allclose(d.coords[0, 0], [0.5, 0.5, 0.5])
