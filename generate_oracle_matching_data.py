"""
Generate oracle matching_data for PuzzleFusion++ evaluation on custom datasets.

This script:
  1. Loads existing converted .npz files (from convert_hdf5_to_npz.py output)
  2. Applies assembly-level normalization so the combined point cloud fits
     in [-0.5, 0.5] — required for the verifier's hardcoded Chamfer-distance
     bins (0, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 100) to fire correctly.
  3. Re-saves normalized .npz files to a `_norm` directory, so both the
     denoiser-only (un-normalized) and full-pipeline (normalized) evaluations
     can coexist for comparison.
  4. Generates oracle matching_data (.npz per sample) using ground-truth
     geometry: points whose nearest neighbor on another fragment is within
     EPS are marked as critical; correspondences pair them across fragments.

Output structure expected by PF++ test.py:
    data/pc_data_norm/<dataset>/val/*.npz    (normalized pc data)
    data/matching_data_oracle/<dataset>/*.npz (per-sample matching data)

Convention (matches Breaking Bad's Jigsaw output):
    edges              : [E, 2] int — each row is [idx2, idx1] (idx2 > idx1)
    correspondence     : [E] object — each entry is [N_matches, 2]:
                           col 0 indexes critical points of part idx1
                           col 1 indexes critical points of part idx2
    gt_pcs             : [num_parts * num_points, 3] float — flattened GT pcs
    critical_pcs_idx   : [num_parts * num_points] int — per-part critical
                           indices, stored contiguously (first n_critical_pcs[i]
                           entries of each per-part block are real, rest 0)
    n_pcs              : [MAX_PARTS] int — points per part (NUM_POINTS or 0)
    n_critical_pcs     : [MAX_PARTS] int — critical points per part

Run with: /home/zhuojiat/.conda/envs/puzzlefusionpp/bin/python generate_oracle_matching_data.py
"""

import os
import sys
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm


# ─── Configuration ───────────────────────────────────────────────────────
PF_ROOT = "/data/gpfs/projects/punim2657/Puzzlefusion"
PC_IN_ROOT = os.path.join(PF_ROOT, "data/pc_data")
PC_OUT_ROOT = os.path.join(PF_ROOT, "data/pc_data_norm")
MD_OUT_ROOT = os.path.join(PF_ROOT, "data/matching_data_oracle")

MAX_PARTS = 20
NUM_POINTS = 1000

# EPS is the absolute distance (in the *normalized* assembly frame where
# the overall bounding box fits in [-0.5, 0.5]) within which a point on
# one fragment is considered to be on the fracture surface shared with
# another fragment. 0.01 ≈ 1% of the overall object extent.
EPS_CRITICAL = 0.02      # threshold for marking a point as "critical"
EPS_CORR = 0.02          # threshold for matching critical points across parts
MIN_EDGE_CORR = 3        # minimum number of correspondence pairs to form an edge


DATASETS = [
    "bone_syn_pig",
    "bone_syn_rib",
    "fractura_real_ceramics",
    "fractura_real_egg",
    "fractura_real_bones",
]


def assembly_normalize(part_pcs_gt, num_parts):
    """Assembly-level normalization: translate+scale so the combined cloud
    fits in [-0.5, 0.5]^3 (matching Breaking Bad convention).

    Args:
        part_pcs_gt: [P, N, 3] raw point clouds in HDF5 frame
        num_parts: int, number of valid parts

    Returns:
        normalized_pcs: [P, N, 3] with same dtype as input
        centroid: [3,] translation applied
        scale: float scale divisor applied (max extent / 1.0)
    """
    valid = part_pcs_gt[:num_parts].reshape(-1, 3)
    bb_min = valid.min(axis=0)
    bb_max = valid.max(axis=0)
    centroid = (bb_min + bb_max) / 2.0
    extent = (bb_max - bb_min).max()  # largest axis extent
    # Scale so largest extent becomes 1.0 (parts fit in [-0.5, 0.5])
    scale = float(extent) if extent > 0 else 1.0

    normalized = part_pcs_gt.copy().astype(np.float32)
    normalized[:num_parts] = (part_pcs_gt[:num_parts] - centroid) / scale
    return normalized, centroid.astype(np.float32), scale


def compute_critical_points(part_pcs, num_parts, eps):
    """For each part, find points that are within `eps` of any other part's
    surface in the assembled (GT) frame.

    Args:
        part_pcs: [P, N, 3] normalized point clouds in GT assembly frame
        num_parts: int
        eps: absolute distance threshold

    Returns:
        critical_indices: list of num_parts arrays; each array contains
                          indices (in [0, N)) of critical points for that part.
    """
    critical = [np.zeros(part_pcs.shape[1], dtype=bool) for _ in range(num_parts)]

    for i in range(num_parts):
        pcs_i = part_pcs[i]
        for j in range(num_parts):
            if i == j:
                continue
            pcs_j = part_pcs[j]
            dists = cdist(pcs_i, pcs_j)  # [N, N]
            nearest = dists.min(axis=1)  # [N]
            critical[i] |= nearest < eps

    return [np.where(m)[0].astype(np.int64) for m in critical]


def compute_edges_and_correspondences(part_pcs, num_parts, critical_indices,
                                      eps_corr, min_edge_corr):
    """Build edge list and per-edge correspondence pairs.

    For a pair of parts (i, j) with i < j, we look for critical points of
    part i whose nearest neighbor among critical points of part j is within
    `eps_corr`. If at least `min_edge_corr` such pairs exist, we add the
    edge [j, i] (following the [idx2, idx1] convention with idx2 > idx1).

    Correspondence columns:
        col 0 -> index into critical_indices[i] (idx1 = i)
        col 1 -> index into critical_indices[j] (idx2 = j)
    """
    edges = []
    correspondences = []

    for i in range(num_parts):
        crits_i = critical_indices[i]
        if len(crits_i) == 0:
            continue
        ci_pts = part_pcs[i][crits_i]  # [Ni, 3]

        for j in range(i + 1, num_parts):
            crits_j = critical_indices[j]
            if len(crits_j) == 0:
                continue
            cj_pts = part_pcs[j][crits_j]  # [Nj, 3]

            d = cdist(ci_pts, cj_pts)  # [Ni, Nj]
            nearest_j = d.argmin(axis=1)
            nearest_d = d.min(axis=1)
            mask = nearest_d < eps_corr
            n_valid = int(mask.sum())
            if n_valid < min_edge_corr:
                continue

            src_idx = np.where(mask)[0].astype(np.int64)  # indices in crits_i
            tgt_idx = nearest_j[mask].astype(np.int64)     # indices in crits_j
            corr = np.column_stack([src_idx, tgt_idx])    # [M, 2]

            # edges convention: [idx2, idx1] with idx2 = j > i = idx1
            edges.append([j, i])
            correspondences.append(corr)

    return edges, correspondences


def pack_matching_data(part_pcs, num_parts, critical_indices,
                       edges, correspondences, num_points=NUM_POINTS):
    """Pack computed data into the .npz schema expected by the verifier.

    gt_pcs layout:          part 0 points | part 1 points | ... | part P-1
                            (total length = num_parts * num_points)
    critical_pcs_idx layout: per part, first n_critical_pcs[i] entries are
                             real (indices in [0, num_points)), remaining
                             entries within that part's slot are 0-padded.
    """
    # Flatten gt_pcs: [P*N, 3]
    gt_pcs = part_pcs[:num_parts].reshape(-1, 3).astype(np.float32)

    # Per-part: NUM_POINTS slots each
    total_len = num_parts * num_points
    critical_pcs_idx = np.zeros(total_len, dtype=np.int64)
    n_pcs = np.zeros(MAX_PARTS, dtype=np.int64)
    n_critical_pcs = np.zeros(MAX_PARTS, dtype=np.int64)

    for i in range(num_parts):
        n_pcs[i] = num_points
        ncrit = len(critical_indices[i])
        n_critical_pcs[i] = ncrit
        start = i * num_points
        critical_pcs_idx[start:start + ncrit] = critical_indices[i]

    # edges array
    if len(edges) == 0:
        edges_arr = np.zeros((0, 2), dtype=np.int64)
    else:
        edges_arr = np.array(edges, dtype=np.int64)

    # correspondence: save as regular int64 [E, N, 2] when all edges have
    # the same number of correspondence pairs (this matches Breaking Bad's
    # convention and works with PyTorch default_collate via .squeeze()).
    # Fall back to object array only when edge correspondence lengths vary.
    if len(correspondences) == 0:
        corr_arr = np.empty((0,), dtype=object)
    else:
        lengths = [len(c) for c in correspondences]
        if all(l == lengths[0] for l in lengths):
            corr_arr = np.stack(correspondences, axis=0).astype(np.int64)
        else:
            corr_arr = np.empty(len(correspondences), dtype=object)
            for idx, c in enumerate(correspondences):
                corr_arr[idx] = c

    return {
        "edges": edges_arr,
        "correspondence": corr_arr,
        "gt_pcs": gt_pcs,
        "critical_pcs_idx": critical_pcs_idx,
        "n_pcs": n_pcs,
        "n_critical_pcs": n_critical_pcs,
    }


def process_sample(in_path, pc_out_path, md_out_path,
                   eps_critical=EPS_CRITICAL,
                   eps_corr=EPS_CORR,
                   min_edge_corr=MIN_EDGE_CORR):
    """Process a single converted .npz: normalize + write matching data.

    Returns summary dict with stats for logging.
    """
    data = np.load(in_path, allow_pickle=True)
    num_parts = int(data["num_parts"])
    part_pcs_gt = data["part_pcs_gt"].astype(np.float32)
    data_id = int(data["data_id"])

    # Handle shape inconsistency: original conversion saved [num_parts, N, 3],
    # not padded to [MAX_PARTS, N, 3]. Pad here for simplicity.
    if part_pcs_gt.shape[0] != MAX_PARTS:
        padded = np.zeros((MAX_PARTS, NUM_POINTS, 3), dtype=np.float32)
        padded[:num_parts] = part_pcs_gt[:num_parts]
        part_pcs_gt = padded

    # 1) Assembly-level normalization
    normalized_pcs, centroid, scale = assembly_normalize(part_pcs_gt, num_parts)

    # 2) Compute oracle critical points + correspondences
    critical_indices = compute_critical_points(
        normalized_pcs, num_parts, eps=eps_critical
    )
    edges, corr = compute_edges_and_correspondences(
        normalized_pcs, num_parts, critical_indices,
        eps_corr=eps_corr, min_edge_corr=min_edge_corr,
    )

    # 3) Pack into matching_data schema
    matching_payload = pack_matching_data(
        normalized_pcs, num_parts, critical_indices,
        edges, corr, num_points=NUM_POINTS
    )

    # 4) Write normalized pc_data .npz
    os.makedirs(os.path.dirname(pc_out_path), exist_ok=True)
    # Preserve the original .npz shape/keys but replace part_pcs_gt
    pc_save = {
        "data_id": data["data_id"],
        "part_valids": data["part_valids"],
        "num_parts": data["num_parts"],
        "mesh_file_path": data["mesh_file_path"],
        "graph": data["graph"],
        "category": data["category"],
        # Keep only the valid parts in the same shape as original
        "part_pcs_gt": normalized_pcs[:num_parts] if data["part_pcs_gt"].shape[0] == num_parts
                       else normalized_pcs,
        "ref_part": data["ref_part"],
    }
    np.savez(pc_out_path, **pc_save)

    # 5) Write matching_data .npz
    os.makedirs(os.path.dirname(md_out_path), exist_ok=True)
    np.savez(md_out_path, **matching_payload)

    return {
        "data_id": data_id,
        "num_parts": num_parts,
        "scale_applied": scale,
        "centroid": centroid,
        "n_edges": len(edges),
        "total_critical": sum(len(c) for c in critical_indices),
        "avg_critical_per_part": (sum(len(c) for c in critical_indices) / max(num_parts, 1)),
        "total_correspondences": sum(len(c) for c in corr),
    }


def process_dataset(dataset_name):
    """Process all samples in one converted dataset directory."""
    in_dir = os.path.join(PC_IN_ROOT, dataset_name, "val")
    pc_out_dir = os.path.join(PC_OUT_ROOT, dataset_name, "val")
    md_out_dir = os.path.join(MD_OUT_ROOT, dataset_name)

    if not os.path.isdir(in_dir):
        print(f"[skip] {dataset_name}: input dir not found ({in_dir})")
        return []

    files = sorted(f for f in os.listdir(in_dir) if f.endswith(".npz"))
    print(f"\n── {dataset_name}: {len(files)} samples ──")
    print(f"  in:  {in_dir}")
    print(f"  out: {pc_out_dir}")
    print(f"       {md_out_dir}")

    results = []
    for fname in tqdm(files, desc=dataset_name):
        in_path = os.path.join(in_dir, fname)
        pc_out_path = os.path.join(pc_out_dir, fname)
        # Matching data is keyed by data_id (integer) — load to get it
        data = np.load(in_path)
        data_id = int(data["data_id"])
        md_out_path = os.path.join(md_out_dir, f"{data_id}.npz")

        stats = process_sample(in_path, pc_out_path, md_out_path)
        results.append(stats)

    return results


def main():
    print("=" * 70)
    print("Oracle matching_data + assembly-normalized pc_data generator")
    print("=" * 70)
    print(f"  EPS_CRITICAL     = {EPS_CRITICAL}")
    print(f"  EPS_CORR         = {EPS_CORR}")
    print(f"  MIN_EDGE_CORR    = {MIN_EDGE_CORR}")
    print(f"  NUM_POINTS       = {NUM_POINTS}")
    print(f"  MAX_PARTS        = {MAX_PARTS}")
    print()

    all_results = {}
    for dname in DATASETS:
        res = process_dataset(dname)
        all_results[dname] = res

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for dname, res in all_results.items():
        if not res:
            print(f"{dname}: skipped")
            continue
        n = len(res)
        avg_edges = np.mean([r["n_edges"] for r in res])
        avg_crit = np.mean([r["avg_critical_per_part"] for r in res])
        avg_corr = np.mean([r["total_correspondences"] for r in res])
        scales = [r["scale_applied"] for r in res]
        print(f"{dname}: {n} samples, "
              f"avg edges/sample={avg_edges:.1f}, "
              f"avg crit_pts/part={avg_crit:.0f}, "
              f"avg corrs/sample={avg_corr:.0f}, "
              f"scale applied: [{min(scales):.4f}, {max(scales):.4f}]")

    print(f"\nOutput:")
    print(f"  Normalized pc_data:  {PC_OUT_ROOT}/<dataset>/val/")
    print(f"  Oracle matching:     {MD_OUT_ROOT}/<dataset>/")


if __name__ == "__main__":
    main()
