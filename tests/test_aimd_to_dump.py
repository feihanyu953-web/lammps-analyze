"""aimd_to_dump regression: vasprun.xml / XDATCAR -> lammpstrj conversion.

Covers the two traps found during implementation (2026-10-22):
- atom type ids must follow first-appearance element order, NOT the atom index
  (an H/O/H system distinguishes: correct [1,2,1], index-bug [1,2,3]);
- vasprun.xml velocities carry no unit declaration: the converter auto-picks
  m/s vs A/fs by kinetic temperature against TEBEG.
Also: triclinic rejection is loud (fail-closed), XDATCAR has no velocity
columns and says so.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parent.parent / "scripts"

V = 2724.7          # m/s per H -> T ~ 300 K for the mini water below


def _mini_vasprun(path: Path, *, triclinic: bool = False) -> None:
    off = " 0.5e-10" if triclinic else " 0.0"
    basis = (f"<v> 0.60e-09{off} 0.0 </v>\n"
             f"<v> 0.0 0.60e-09 0.0 </v>\n"
             f"<v> 0.0 0.0 0.60e-09 </v>\n")
    calc = []
    for k in range(2):
        pos = (f"<v> 0.125 0.5 0.5 </v>\n<v> 0.5 0.5 0.5 </v>\n"
               f"<v> 0.374 0.376 0.5 </v>\n")
        vel = (f"<v> {V} 0.0 0.0 </v>\n<v> 0.0 0.0 0.0 </v>\n"
               f"<v> {-V} 0.0 0.0 </v>\n")
        calc.append(
            " <calculation>\n  <structure>\n   <crystal>\n"
            f"    <varray name=\"basis\" >\n{basis}    </varray>\n"
            "   </crystal>\n"
            f"    <varray name=\"positions\" >\n{pos}    </varray>\n"
            "  </structure>\n"
            f"  <varray name=\"velocities\" >\n{vel}  </varray>\n"
            " </calculation>\n")
    xml = f"""<?xml version="1.0" encoding="ISO-8859-1"?>
<modeling>
 <atominfo>
  <atoms>       3 </atoms>
  <types>       2 </types>
  <array name="atoms" >
   <field type="string">element</field>
   <field type="int">atomtype</field>
   <set comment="atom types">
    <rc><c>H</c><c>       1</c></rc>
    <rc><c>O</c><c>       2</c></rc>
    <rc><c>H</c><c>       1</c></rc>
   </set>
  </array>
  <array name="atomtypes" >
   <field type="int">atomspertype</field>
   <field type="float">mass</field>
   <field type="float">valence</field>
   <field type="string">element</field>
   <set comment="atom types">
    <rc><c>       2</c><c>     1.00800000</c><c>     1.00000000</c><c>H</c></rc>
    <rc><c>       1</c><c>    15.99900000</c><c>     6.00000000</c><c>O</c></rc>
   </set>
  </array>
 </atominfo>
 <parameters>
  <i name="TEBEG">      300.00</i>
 </parameters>
{''.join(calc)}</modeling>
"""
    path.write_text(xml, encoding="utf-8")


def test_vasprun_conversion_types_and_velocity_calibration(tmp_path):
    src = tmp_path / "vasprun.xml"
    _mini_vasprun(src)
    out = tmp_path / "traj.lammpstrj"
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "aimd_to_dump.py"),
         "--vasprun", str(src), "--out", str(out), "--timestep-fs", "1.0"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "interpreted as 'ms'" in r.stdout
    assert "T~300" in r.stdout.replace(" ", "") or "T~300" in r.stdout

    # type ids by first-appearance element order (H=1, O=2), NOT atom index:
    # the third atom is H and must carry type 1, not 3
    body = out.read_text().splitlines()
    types_frame0 = [ln.split()[1] for ln in body
                    if len(ln.split()) >= 5 and ln.split()[0] in ("1", "2", "3")
                    and "ITEM" not in ln][:3]
    assert types_frame0 == ["1", "2", "1"]
    assert "ITEM: ATOMS id type x y z vx vy vz" in body   # velocities carried
    assert (tmp_path / "traj.elements_map").read_text().strip() == "1:H 2:O"

    sys.path.insert(0, str(SCRIPTS))
    from dump_io import parse_dump
    d = parse_dump(out, timestep_fs=1.0)
    assert d.natoms == 3 and len(d.timesteps) == 2
    assert d.wrapped is False or d.columns[2].startswith("x")
    assert d.cell[0, 0, 0] == pytest.approx(6.0)          # meters -> A
    assert d.velocities is not None                       # VACF/VDOS-capable
    # MDAnalysis compatibility: exactly ONE leading comment line allowed
    # (two comment lines broke l2_rdf_cn on the real-data end-to-end test)
    import MDAnalysis as mda
    u = mda.Universe(str(out), format="LAMMPSDUMP")
    assert len(u.atoms) == 3


def test_vasprun_triclinic_rejected_loudly(tmp_path):
    src = tmp_path / "vasprun_tri.xml"
    _mini_vasprun(src, triclinic=True)
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "aimd_to_dump.py"),
         "--vasprun", str(src), "--out", str(tmp_path / "x.lammpstrj")],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "triclinic" in (r.stderr + r.stdout)


def test_xdatcar_conversion(tmp_path):
    src = tmp_path / "XDATCAR"
    src.write_text(
        "NaCl mini\n"
        "1.0\n"
        " 5.0 0.0 0.0\n"
        " 0.0 5.0 0.0\n"
        " 0.0 0.0 5.0\n"
        "Na Cl\n"
        "1 1\n"
        "Direct configuration=     0\n"
        " 0.0 0.0 0.0\n"
        " 0.5 0.5 0.5\n"
        "Direct configuration=     1\n"
        " 0.01 0.0 0.0\n"
        " 0.51 0.5 0.5\n", encoding="utf-8")
    out = tmp_path / "xdat.lammpstrj"
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "aimd_to_dump.py"),
         "--xdatcar", str(src), "--out", str(out), "--timestep-fs", "2.0"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    body = out.read_text().splitlines()
    assert "ITEM: ATOMS id type x y z\n" in out.read_text()  # no velocities
    assert "VACF/VDOS impossible" in r.stdout
    types = [ln.split()[1] for ln in body
             if len(ln.split()) >= 5 and ln.split()[0] in ("1", "2")
             and "ITEM" not in ln][:2]
    assert types == ["1", "2"]                # Na=1, Cl=2 by first appearance
    # Cartesian check: fractional 0.5 on a 5 A cell -> 2.5 A
    coords = [ln.split() for ln in body
              if len(ln.split()) >= 5 and ln.split()[0] == "2"][:1]
    assert np.allclose([float(x) for x in coords[0][2:5]], [2.5, 2.5, 2.5])
