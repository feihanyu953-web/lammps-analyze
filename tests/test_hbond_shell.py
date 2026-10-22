"""l2_hbond_shell regression: shell classification follows a MOVING ion.

Hand-checkable 2-frame system (box 20 A, types 1=O 2=H 3=K):
  frame 0: K at (5,5,5); water A (O 7,5,5; H 7.96,5,5) -- in shell (K-O 2.0 A),
           H_A points at water B (O 9.5,5,5; H 10.1,5,5) -- bulk (K-O 4.5 A).
           Exactly one hbond A(donor)->B: shell set {A}, touch A=1, B=1.
  frame 1: K moved to (15,15,15) -> shell empty; the SAME hbond is now
           classified bulk_bulk. Shell membership follows the ion.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).parent.parent / "scripts"


def _write_dump(path: Path) -> None:
    frames = [
        # frame 0: K near water A
        [(1, 3, 5.0, 5.0, 5.0), (2, 1, 7.0, 5.0, 5.0), (3, 2, 7.96, 5.0, 5.0),
         (4, 1, 9.5, 5.0, 5.0), (5, 2, 10.1, 5.0, 5.0)],
        # frame 1: K far away
        [(1, 3, 15.0, 15.0, 15.0), (2, 1, 7.0, 5.0, 5.0), (3, 2, 7.96, 5.0, 5.0),
         (4, 1, 9.5, 5.0, 5.0), (5, 2, 10.1, 5.0, 5.0)],
    ]
    with open(path, "w", encoding="utf-8") as fh:
        for i, atoms in enumerate(frames):
            fh.write(f"ITEM: TIMESTEP\n{i}\nITEM: NUMBER OF ATOMS\n5\n"
                     "ITEM: BOX BOUNDS pp pp pp\n0 20\n0 20\n0 20\n"
                     "ITEM: ATOMS id type x y z\n")
            for aid, t, x, y, z in atoms:
                fh.write(f"{aid} {t} {x} {y} {z}\n")


def test_shell_classification_follows_moving_ion(tmp_path):
    dump = tmp_path / "mini.dump"
    _write_dump(dump)
    outdir = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "l2_hbond_shell.py"),
         "--dump", str(dump), "--elements-map", "1:O 2:H 3:K",
         "--shell-around", "K", "--shell-rcut", "3.5",
         "--name", "t", "--outdir", str(outdir)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    lines = (outdir / "t_hbshell_counts.csv").read_text().strip().split("\n")
    assert lines[0] == "frame,n_shell,hb_per_shell,hb_per_bulk"
    f0, f1 = lines[1].split(","), lines[2].split(",")
    assert (f0[1], float(f0[2]), float(f0[3])) == ("1", 1.0, 1.0)
    assert f1[1] == "0"                       # shell emptied
    assert np.isnan(float(f1[2]))             # no shell waters -> nan
    assert float(f1[3]) == 1.0                # hbond persists, now bulk

    ct = (outdir / "t_hbshell_ct.csv").read_text().strip().split("\n")
    assert ct[0] == "frame,C_shell_shell,C_donor_shell,C_bulk_bulk"
    c0 = ct[1].split(",")
    assert float(c0[1]) == 0.0                # no shell_shell events at all
    assert float(c0[2]) == 1.0                # C(0)=1: donor_shell run exists
    assert float(c0[3]) == 1.0                # C(0)=1: bulk_bulk run exists
    c1 = ct[2].split(",")
    assert float(c1[2]) == 0.0                # frame-1 event gone (run L=1)
    assert float(c1[3]) == 0.0

    assert "ratio 1.00" in r.stdout           # 1.0 vs 1.0 on this tiny system
    pngs = sorted(p.name for p in outdir.glob("*.png"))
    assert pngs == ["t_hbshell_counts.png", "t_hbshell_ct.png"]
