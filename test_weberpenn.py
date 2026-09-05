"""Grows the paper's four trees at two seeds and checks them against weber-penn-net,
whose fingerprints are in expected.txt: stems per level, sections, leaves, height,
width, sums of positions and radii, and the mesh. Run with `python3 test_weberpenn.py`."""
import json
from pathlib import Path

from weberpenn import Mesh, Parameters, Tree

HERE = Path(__file__).parent


def fingerprint(name, table, seed):
    t = Tree.grow(Parameters.from_paper(table), seed)
    sx = sy = sz = sr = 0.0
    n = 0
    for s in t.stems:
        for q in s.sections:
            sx += q.position.x
            sy += q.position.y
            sz += q.position.z
            sr += q.radius
            n += 1
    lx = sum(l.frame.origin.x + l.frame.origin.z for l in t.leaves)
    m = Mesh.of(t, 6, 3)
    mx = sum(v.x + v.y + v.z for v in m.vertices)
    counts = {}
    for s in t.stems:
        counts[s.level] = counts.get(s.level, 0) + 1
    levels = " ".join(f"L{k}={v}" for k, v in sorted(counts.items()))
    return f"{name} seed {seed}: {levels} sections {n} leaves {len(t.leaves)} height {t.height:.6f} width {t.width:.6f} pos {sx:.4f} {sy:.4f} {sz:.4f} r {sr:.6f} leafsum {lx:.4f} mesh {len(m.vertices)} {len(m.faces)} {mx:.4f}"


def main():
    presets = json.loads((HERE / "presets.json").read_text())
    expected = (HERE / "expected.txt").read_text().splitlines()
    got = [fingerprint(name, table, seed) for name, table in presets.items() for seed in (13, 7)]
    for want, have in zip(expected, got):
        assert want == have, f"\n  want {want}\n  have {have}"
    print(f"{len(got)} trees the same as weber-penn-net")


if __name__ == "__main__":
    main()
