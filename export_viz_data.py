"""
Export visualization payload for the interactive PuzzleFusion++ viewer.

This script produces a compact `viz_data.json` with:
- downsampled, centroid-centered base points per fragment
- rigid-body transforms (pos + quat) for:
  - scattered layout
  - ground-truth assembly
  - final predicted assembly
  - selected denoising keyframes
- mesh metadata (relative path + fragment `.obj` filenames) for mesh mode.
"""

import glob
import json
import os
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


PROJDIR = "/data/gpfs/projects/punim2657/Puzzlefusion"
VAL_DIR = os.path.join(PROJDIR, "data/pc_data/everyday/val/")
INFER_DIR = os.path.join(
    PROJDIR, "output/denoiser/everyday_epoch2000_bs64/inference/test_run"
)
BB_ROOT = "/data/gpfs/projects/punim2657/Breaking-Bad-Dataset.github.io/data/breaking_bad"

TARGET_DATA_ID = 97
PTS_PER_FRAG = 60  # base points per fragment for web rendering


def fps_downsample(pts: np.ndarray, k: int) -> np.ndarray:
    """Simple farthest-point sampling for roughly uniform coverage."""
    if pts.shape[0] <= k:
        return pts

    selected = [0]
    dists = np.full(pts.shape[0], np.inf)
    for _ in range(k - 1):
        d = np.linalg.norm(pts - pts[selected[-1]], axis=1)
        dists = np.minimum(dists, d)
        selected.append(int(np.argmax(dists)))
    return pts[selected]


def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angular error in degrees between two scalar-first [w,x,y,z] quaternions."""
    q1s = q1[[1, 2, 3, 0]]  # scipy expects [x, y, z, w]
    q2s = q2[[1, 2, 3, 0]]
    rel = R.from_quat(q1s) * R.from_quat(q2s).inv()
    deg = float(rel.magnitude() * 180.0 / np.pi)
    return min(deg, 360.0 - deg)


def load_inference() -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    """Load gt / predicted trajectories and mesh path for a single data_id."""
    gt_saved = np.load(
        os.path.join(INFER_DIR, str(TARGET_DATA_ID), "gt.npy")
    )  # (n, 7)

    pfiles = glob.glob(
        os.path.join(INFER_DIR, str(TARGET_DATA_ID), "predict_*.npy")
    )
    assert pfiles, f"No predict file found for data_id={TARGET_DATA_ID}"
    # There is only one predict file per sample; name encodes part_acc.
    pfile = pfiles[0]
    pred_acc = float(
        os.path.basename(pfile).split("predict_")[1].replace(".npy", "")
    )
    pred_all = np.load(pfile)  # (T, n, 7)

    with open(
        os.path.join(INFER_DIR, str(TARGET_DATA_ID), "mesh_file_path.txt")
    ) as f:
        mesh_path = f.read().strip()

    return gt_saved, pred_all, gt_saved.copy(), mesh_path, pred_acc


def load_val_pcs(n_parts: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load canonical part_pcs_gt and ref_part for TARGET_DATA_ID from val split."""
    pcs_gt = None
    ref_part = None
    for fname in sorted(os.listdir(VAL_DIR)):
        d = np.load(os.path.join(VAL_DIR, fname), allow_pickle=True)
        if int(d["data_id"].item()) == TARGET_DATA_ID:
            pcs_gt = d["part_pcs_gt"][:n_parts]  # (n, 1000, 3)
            ref_part = d["ref_part"][:n_parts]
            print(f"Loaded canonical pcs from {fname}")
            break

    assert pcs_gt is not None, f"Could not find val sample with data_id={TARGET_DATA_ID}"
    return pcs_gt.astype(np.float64), ref_part.astype(bool)


def compute_transforms(
    gt_saved: np.ndarray, pred_all: np.ndarray
) -> Tuple[List[float], List[float], List[float]]:
    """Compute per-fragment + per-step rotation and translation errors."""
    T, n_parts = pred_all.shape[0], pred_all.shape[1]

    # Final-step per-fragment errors
    rot_errors = [
        quat_angle_deg(pred_all[-1][i][3:], gt_saved[i][3:])
        for i in range(n_parts)
    ]
    trans_errors = [
        float(np.linalg.norm(pred_all[-1][i][:3] - gt_saved[i][:3]))
        for i in range(n_parts)
    ]

    # Mean rotation error trajectory
    traj_mean_rot_err: List[float] = []
    for t in range(T):
        errs_t = [
            quat_angle_deg(pred_all[t][i][3:], gt_saved[i][3:])
            for i in range(n_parts)
        ]
        traj_mean_rot_err.append(float(np.mean(errs_t)))

    return rot_errors, trans_errors, traj_mean_rot_err


def build_payload() -> None:
    gt_saved, pred_all, _, mesh_path, pred_acc = load_inference()
    T, n_parts = pred_all.shape[0], pred_all.shape[1]

    print(f"Loaded inference data: {n_parts} parts, {T} denoising steps")
    print(f"Sample: {mesh_path}")

    pcs_gt, ref_part = load_val_pcs(n_parts)

    # Canonical centroids in assembled frame.
    centroids_gt = np.array([pcs_gt[i].mean(axis=0) for i in range(n_parts)])

    # Centered base geometry per fragment (for both points + mesh transforms).
    pcs_centered = [pcs_gt[i] - centroids_gt[i] for i in range(n_parts)]
    base_points = [fps_downsample(p, PTS_PER_FRAG) for p in pcs_centered]

    # Scatter layout grid.
    scene_scale = float(np.max(np.abs(pcs_gt)))
    grid_spacing = scene_scale * 2.0
    cols = int(np.ceil(np.sqrt(n_parts)))
    grid_offsets: List[np.ndarray] = []
    for i in range(n_parts):
        row, col = divmod(i, cols)
        grid_offsets.append(np.array([col * grid_spacing, row * grid_spacing, 0.0]))

    # Helper: compose transform arrays [tx,ty,tz,qw,qx,qy,qz].
    def make_transform(pos: np.ndarray, quat: np.ndarray) -> List[float]:
        return [
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(quat[0]),
            float(quat[1]),
            float(quat[2]),
            float(quat[3]),
        ]

    # Identity quaternion (scalar-first).
    q_identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # Scattered transforms: centered geometry placed on grid with identity rotations.
    scattered_tf = [make_transform(grid_offsets[i], q_identity) for i in range(n_parts)]

    # Ground-truth transforms: centered geometry placed at canonical centroids.
    gt_tf = [make_transform(centroids_gt[i], q_identity) for i in range(n_parts)]

    # Final predicted transforms: relative rotation error + translation displacement
    anchor_idx = int(np.where(ref_part)[0][0])
    anchor_centroid = centroids_gt[anchor_idx]

    pred_final_tf: List[List[float]] = []
    for i in range(n_parts):
        if i == anchor_idx:
            pred_final_tf.append(make_transform(centroids_gt[i], q_identity))
            continue

        q_gt = gt_saved[i][3:]  # [w,x,y,z]
        q_pred = pred_all[-1][i][3:]
        t_gt = gt_saved[i][:3]
        t_pred = pred_all[-1][i][:3]

        R_gt = R.from_quat(q_gt[[1, 2, 3, 0]])
        R_pred = R.from_quat(q_pred[[1, 2, 3, 0]])
        R_rel = R_pred * R_gt.inv()
        q_rel = R_rel.as_quat()  # [x,y,z,w]
        q_rel = q_rel[[3, 0, 1, 2]]  # scalar-first

        t_disp = t_pred - t_gt
        t_norm = np.linalg.norm(t_gt) or 1.0
        canon_norm = np.linalg.norm(centroids_gt[i] - anchor_centroid) or 1e-6
        scale_factor = canon_norm / (t_norm + 1e-6) * 0.3
        t_visual = t_disp * scale_factor
        pos = centroids_gt[i] + t_visual

        pred_final_tf.append(make_transform(pos, q_rel))

    # Denoising trajectory transforms (keyframes).
    stride = max(1, T // 12)
    traj_steps: List[int] = list(range(0, T, stride))
    if T - 1 not in traj_steps:
        traj_steps.append(T - 1)

    traj_keyframes: List[List[List[float]]] = []
    for t_idx in traj_steps:
        frame_tf: List[List[float]] = []
        for i in range(n_parts):
            if i == anchor_idx:
                frame_tf.append(make_transform(centroids_gt[i], q_identity))
                continue

            q_gt = gt_saved[i][3:]
            q_pred = pred_all[t_idx][i][3:]
            t_gt = gt_saved[i][:3]
            t_pred = pred_all[t_idx][i][:3]

            R_gt = R.from_quat(q_gt[[1, 2, 3, 0]])
            R_pred = R.from_quat(q_pred[[1, 2, 3, 0]])
            R_rel = R_pred * R_gt.inv()
            q_rel = R_rel.as_quat()
            q_rel = q_rel[[3, 0, 1, 2]]

            t_disp = t_pred - t_gt
            t_norm = np.linalg.norm(t_gt) or 1.0
            canon_norm = np.linalg.norm(centroids_gt[i] - anchor_centroid) or 1e-6
            scale_factor = canon_norm / (t_norm + 1e-6) * 0.3
            t_visual = t_disp * scale_factor
            pos = centroids_gt[i] + t_visual

            frame_tf.append(make_transform(pos, q_rel))

        traj_keyframes.append(frame_tf)

    rot_errors, trans_errors, traj_mean_rot_err = compute_transforms(
        gt_saved, pred_all
    )

    print(
        f"\nPer-fragment rotation errors (°): "
        f"min={min(rot_errors):.1f}, max={max(rot_errors):.1f}, "
        f"mean={np.mean(rot_errors):.1f}"
    )
    print(
        f"Per-fragment translation errors:  "
        f"min={min(trans_errors):.3f}, max={max(trans_errors):.3f}"
    )
    print(
        f"\nDenoising trajectory — mean rot error: "
        f"step 0: {traj_mean_rot_err[0]:.1f}°  →  final: {traj_mean_rot_err[-1]:.1f}°"
    )

    # Mesh fragment file list.
    mesh_dir = os.path.join(BB_ROOT, mesh_path)
    obj_files = sorted(
        f for f in os.listdir(mesh_dir) if f.lower().endswith(".obj")
    )
    print(f"Found {len(obj_files)} mesh fragments in {mesh_dir}")

    COLORS = [
        "#E53935",
        "#1E88E5",
        "#43A047",
        "#FB8C00",
        "#8E24AA",
        "#00ACC1",
        "#E040FB",
        "#F4511E",
        "#7CB342",
        "#3949AB",
        "#00897B",
        "#FFB300",
        "#D81B60",
        "#6D4C41",
        "#546E7A",
        "#C0CA33",
        "#039BE5",
        "#F06292",
        "#80CBC4",
        "#FFF176",
    ]

    print("\nBuilding JSON payload...")

    payload = {
        "n_parts": n_parts,
        "anchor_idx": int(anchor_idx),
        "mesh_path": mesh_path,
        "obj_files": obj_files,
        "part_acc": float(pred_acc),
        "colors": COLORS[:n_parts],
        "rot_errors": rot_errors,
        "trans_errors": trans_errors,
        "traj_mean_rot_err": traj_mean_rot_err,
        "traj_steps": traj_steps,
        "T": int(T),
        # geometry
        "base_points": [p.tolist() for p in base_points],
        # transforms for modes
        "scattered_tf": scattered_tf,
        "gt_tf": gt_tf,
        "pred_tf": pred_final_tf,
        "traj_tf": traj_keyframes,
    }

    out_path = os.path.join(PROJDIR, "viz_data.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Saved viz_data.json ({size_mb:.1f} MB)")


if __name__ == "__main__":
    build_payload()
