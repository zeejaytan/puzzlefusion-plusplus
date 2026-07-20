#!/usr/bin/env python3
"""T2b — VQVAE representation invariance probe (mirror of GARF Exp 10).

Encodes each part of two npz sets (same object, different fracture-surface
wear, same scan frame) with the frozen PF++ VQVAE encoder and reports, per
part: cosine similarity of z_q (25x64) and the fraction of identical codebook
token indices (100 x 16D tokens). A cross-part null (different sherds, same
set) calibrates what "different shapes" look like.

Gate (plan T2b): token overlap >= 0.8 and same-part similarity far above the
cross-part null => the representation ignores the wear manipulation that
moved GARF's fracture response 3.8x (Exp 14).

Usage (PF++ repo root, puzzlefusionpp env):
  python scripts/vqvae_latent_probe.py \
      --npz-a data/pc_data/juglet_deploy/val/00000.npz \
      --npz-b data/pc_data/juglet_dewear/val/00000.npz \
      --ckpt output/denoiser/everyday_epoch2000_bs64/training/last.ckpt \
      --out  logs/t2b_latent_probe
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize

from puzzlefusion_plusplus.vqvae.model.modules.vq_vae import VQVAE


def normalize_parts(pcs):
    """Dataset-style per-part normalisation, minus the random rotation."""
    out, scales = [], []
    for pc in pcs:
        c = pc - pc.mean(0)
        s = np.max(np.abs(c))
        out.append(c / (s if s > 0 else 1.0))
        scales.append(s)
    return np.stack(out), np.array(scales)


def token_indices(model, z_e):
    """z_e (B,25,64) -> codebook indices (B,100)."""
    B, L, C = z_e.shape
    e_dim = model.vector_quantization.e_dim
    z = z_e.reshape(B, L * (C // e_dim), e_dim).reshape(-1, e_dim)
    w = model.vector_quantization.embedding.weight
    d = (z.pow(2).sum(1, keepdim=True) + w.pow(2).sum(1)
         - 2 * z @ w.t())
    return d.argmin(1).reshape(B, -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz-a", type=Path, required=True)
    ap.add_argument("--npz-b", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    with initialize(config_path="../config", version_base=None):
        cfg = compose(config_name="auto_aggl")
    model = VQVAE(cfg.ae)
    sd = torch.load(args.ckpt, map_location="cpu")["state_dict"]
    model.load_state_dict({k.replace("encoder.", "", 1): v
                           for k, v in sd.items() if k.startswith("encoder.")})
    model = model.to(dev).eval()

    pcs_a = np.load(args.npz_a, allow_pickle=True)["part_pcs_gt"].astype(np.float32)
    pcs_b = np.load(args.npz_b, allow_pickle=True)["part_pcs_gt"].astype(np.float32)
    assert pcs_a.shape[0] == pcs_b.shape[0]
    na, _ = normalize_parts(pcs_a)
    nb, _ = normalize_parts(pcs_b)

    with torch.no_grad():
        ta = torch.from_numpy(na).to(dev)
        tb = torch.from_numpy(nb).to(dev)
        za, xa = model.encoder(ta.permute(0, 2, 1))  # z_e (P,25,64), xyz (P,25,3)
        zb, xb = model.encoder(tb.permute(0, 2, 1))
        ia = token_indices(model, za).reshape(za.shape[0], 25, -1)
        ib = token_indices(model, zb).reshape(zb.shape[0], 25, -1)
        _, qa, _, _, _ = model.vector_quantization(za.reshape(za.shape[0], -1, 16))
        _, qb, _, _, _ = model.vector_quantization(zb.reshape(zb.shape[0], -1, 16))
        qa = qa.reshape(qa.shape[0], 25, -1).cpu()  # (P,25,64)
        qb = qb.reshape(qb.shape[0], 25, -1).cpu()
        xa, xb = xa.cpu().numpy(), xb.cpu().numpy()
        ia, ib = ia.cpu(), ib.cpu()

    # FPS center sets differ between encodings of resampled clouds, so the
    # 25 latents have no index correspondence — match by nearest center.
    from scipy.spatial import cKDTree
    P = qa.shape[0]
    cos = torch.nn.functional.cosine_similarity

    def matched_stats(q1, i1, x1, q2, i2, x2):
        """Match centers of set1 -> nearest center of set2; compare latents."""
        idx = cKDTree(x2).query(x1)[1]
        cdist = float(np.linalg.norm(x2[idx] - x1, axis=1).mean())
        c = float(cos(q1.reshape(-1), q2[idx].reshape(-1), dim=0))
        tok = float((i1 == i2[idx]).float().mean())
        return c, tok, cdist

    rows = []
    for i in range(P):
        c, tok, cd = matched_stats(qa[i], ia[i], xa[i], qb[i], ib[i], xb[i])
        rows.append({"part": i, "cosine_same_part": c, "token_overlap": tok,
                     "center_match_dist": cd})
    # cross-part null within set A (matched the same way)
    null_cos, null_tok = [], []
    for i in range(P):
        for j in range(P):
            if i != j:
                c, tok, _ = matched_stats(qa[i], ia[i], xa[i],
                                          qa[j], ia[j], xa[j])
                null_cos.append(c)
                null_tok.append(tok)

    summary = {
        "same_part_cosine": {"mean": float(np.mean([r["cosine_same_part"] for r in rows])),
                             "min": float(np.min([r["cosine_same_part"] for r in rows]))},
        "same_part_token_overlap": {"mean": float(np.mean([r["token_overlap"] for r in rows])),
                                    "min": float(np.min([r["token_overlap"] for r in rows]))},
        "cross_part_null": {"cosine_mean": float(np.mean(null_cos)),
                            "token_overlap_mean": float(np.mean(null_tok))},
        "gate_token_overlap_ge_0.8":
            float(np.mean([r["token_overlap"] for r in rows])) >= 0.8,
        "per_part": rows,
    }
    with open(args.out / "latent_probe.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "per_part"},
                     indent=2))
    print("per part:")
    for r in rows:
        print(f"  part {r['part']}: cos {r['cosine_same_part']:.4f}  "
              f"token overlap {r['token_overlap']:.3f}")
    print(f"wrote {args.out}/latent_probe.json")


if __name__ == "__main__":
    main()
