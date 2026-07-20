#!/usr/bin/env python3
"""T3 — build 2-piece PF++ npz datasets for the pairwise mating oracle.

juglet mode: all C(9,2)=36 pairs from the Juglet deploy npz (scan frame).
control mode: every pair sample in GARF's control_ceramics_pairs.hdf5
  (pieces stored in assembled GT coords — those coords double as the scoring
  reference).

Each output sample: part_pcs_gt (2,1000,3), part_valids/ref_part padded to 20,
graph 20x20 with the (0,1) edge, ref part = larger piece by bbox diagonal.
A manifest.json maps data_id -> pair name.

Usage:
  python scripts/build_pair_npz.py juglet \
      --base data/pc_data/juglet_deploy/val/00000.npz \
      --out data/pc_data/t3_juglet_pairs
  python scripts/build_pair_npz.py control \
      --hdf5 /data/gpfs/projects/punim2657/GARF/input/control_ceramics_pairs.hdf5 \
      --out data/pc_data/t3_control_pairs
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

MAX_PARTS = 20


def diag(pc):
    return float(np.linalg.norm(pc.max(0) - pc.min(0)))


def write_sample(out_val: Path, data_id: int, pc0, pc1, mesh_path: str,
                 category: str):
    valids = np.zeros(MAX_PARTS, dtype=np.float32)
    valids[:2] = 1
    ref = np.zeros(MAX_PARTS, dtype=bool)
    ref[0 if diag(pc0) >= diag(pc1) else 1] = True
    graph = np.zeros((MAX_PARTS, MAX_PARTS), dtype=bool)
    graph[0, 1] = graph[1, 0] = True
    np.savez(
        out_val / f"{data_id:05d}.npz",
        data_id=np.int64(data_id),
        part_valids=valids,
        num_parts=np.int64(2),
        mesh_file_path=mesh_path,
        graph=graph,
        category=category,
        part_pcs_gt=np.stack([pc0, pc1], axis=0).astype(np.float64),
        ref_part=ref,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["juglet", "control"])
    ap.add_argument("--base", type=Path, help="juglet deploy npz")
    ap.add_argument("--hdf5", type=Path, help="control pairs hdf5")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-points", type=int, default=1000)
    args = ap.parse_args()
    out_val = args.out / "val"
    out_val.mkdir(parents=True, exist_ok=True)
    manifest = {}

    if args.mode == "juglet":
        base = np.load(args.base, allow_pickle=True)
        pcs = base["part_pcs_gt"].astype(np.float64)
        P = pcs.shape[0]
        for did, (i, j) in enumerate(combinations(range(P), 2)):
            write_sample(out_val, did, pcs[i], pcs[j],
                         str(base["mesh_file_path"].item()), "artifact")
            manifest[did] = {"pair": f"p{i+1:02d}{j+1:02d}", "i": i, "j": j}
    else:
        import h5py
        import trimesh
        hf = h5py.File(args.hdf5, "r")
        names = sorted(hf["control"].keys())
        for did, name in enumerate(names):
            g = hf["control"][name]["pieces"]
            pts = []
            for k in ("0", "1"):
                m = trimesh.Trimesh(
                    vertices=np.asarray(g[k]["vertices"]),
                    faces=np.asarray(g[k]["faces"]), process=False)
                p, _ = trimesh.sample.sample_surface(m, args.n_points)
                pts.append(np.asarray(p, dtype=np.float64))
            write_sample(out_val, did, pts[0], pts[1], f"control/{name}",
                         "control")
            manifest[did] = {"pair": name}
        hf.close()

    with open(args.out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(manifest)} pair samples to {out_val} + manifest.json")


if __name__ == "__main__":
    main()
