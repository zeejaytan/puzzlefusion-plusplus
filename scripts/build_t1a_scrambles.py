#!/usr/bin/env python3
"""T1a — build sherd-identity scramble variants of the Juglet deploy npz.

Arms (PFPP_JUGLET_SUCCESS_EXPERIMENT_PLAN.md):
  foreign — replace the 4 largest non-anchor sherds with sherds from
            fractura_real_ceramics (different vessels), each rescaled to the
            replaced sherd's bbox diagonal and recentred at its scan centroid
            (size held constant, shape identity swapped)
  dup     — every slot gets a copy of one mid-size non-anchor body sherd
            (zero mating structure, scale distribution collapsed to one value)
  mirror  — every sherd mirrored about its centroid (x -> -x): coarse shape
            statistics preserved, mating geometrically impossible

Usage:
  python scripts/build_t1a_scrambles.py \
      --base data/pc_data/juglet_deploy/val/00000.npz \
      --foreign-dir data/pc_data/fractura_real_ceramics/val \
      --out-root data/pc_data
"""

import argparse
import json
from pathlib import Path

import numpy as np


def diag(pc):
    return float(np.linalg.norm(pc.max(0) - pc.min(0)))


def save_variant(base, pcs, out_dir: Path, note: dict):
    out_dir.joinpath("val").mkdir(parents=True, exist_ok=True)
    out = {k: base[k] for k in base.files}
    out["part_pcs_gt"] = np.stack(pcs, axis=0).astype(np.float64)
    np.savez(out_dir / "val" / "00000.npz", **out)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(note, f, indent=2)
    print(f"wrote {out_dir}/val/00000.npz  ({note['arm']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--foreign-dir", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--n-foreign", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = np.load(args.base, allow_pickle=True)
    pcs = base["part_pcs_gt"].astype(np.float64)
    P = pcs.shape[0]
    ref_idx = int(np.where(base["ref_part"][:P])[0][0])
    diags = [diag(pcs[i]) for i in range(P)]
    print(f"{P} parts, anchor index {ref_idx}, diags "
          + " ".join(f"{d:.3f}" for d in diags))

    # ---- foreign
    non_ref = [i for i in range(P) if i != ref_idx]
    targets = sorted(non_ref, key=lambda i: -diags[i])[: args.n_foreign]
    ffiles = sorted(args.foreign_dir.glob("*.npz"))
    rng = np.random.default_rng(args.seed)
    foreign_srcs = []
    for f in ffiles:
        d = np.load(f, allow_pickle=True)
        fp = d["part_pcs_gt"].astype(np.float64)
        fref = int(np.where(d["ref_part"][: fp.shape[0]])[0][0])
        cands = [k for k in range(fp.shape[0]) if k != fref]
        if cands:
            foreign_srcs.append((f.name, int(rng.choice(cands)), fp))
        if len(foreign_srcs) >= args.n_foreign:
            break
    assert len(foreign_srcs) >= args.n_foreign, "not enough foreign sherds"
    pcs_f = [pcs[i].copy() for i in range(P)]
    note = {"arm": "foreign", "anchor": ref_idx, "replaced": {}}
    for t, (fname, k, fp) in zip(targets, foreign_srcs):
        src = fp[k]
        src_c = src - src.mean(0)
        src_c *= diags[t] / (diag(src) + 1e-12)
        pcs_f[t] = src_c + pcs[t].mean(0)
        note["replaced"][str(t)] = f"{fname}:part{k}"
    save_variant(base, pcs_f, args.out_root / "t1a_foreign", note)

    # ---- dup
    body = sorted(non_ref, key=lambda i: diags[i])[len(non_ref) // 2]
    pcs_d = []
    for i in range(P):
        src = pcs[body] - pcs[body].mean(0)
        pcs_d.append(src + pcs[i].mean(0))
    save_variant(base, pcs_d, args.out_root / "t1a_dup",
                 {"arm": "dup", "source_part": body, "anchor": ref_idx})

    # ---- mirror
    pcs_m = []
    for i in range(P):
        c = pcs[i].mean(0)
        m = pcs[i] - c
        m[:, 0] *= -1
        pcs_m.append(m + c)
    save_variant(base, pcs_m, args.out_root / "t1a_mirror",
                 {"arm": "mirror", "anchor": ref_idx})


if __name__ == "__main__":
    main()
