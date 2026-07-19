"""
Convert GARF HDF5 datasets to PuzzleFusion++ .npz format.

This script reads mesh data (vertices + faces) from GARF-format HDF5 files,
samples point clouds, computes connectivity graphs, and saves in the .npz
format expected by PuzzleFusion++'s GeometryLatentDataset.

Usage (run with GARF venv that has h5py + trimesh):
    python convert_hdf5_to_npz.py

Single HDF5 (e.g. archaeological deploy):

    python convert_hdf5_to_npz.py \\
      --hdf5 /path/to/juglet_deploy.hdf5 \\
      --category artifact \\
      --output-dir /path/to/Puzzlefusion/data/pc_data/juglet_deploy/val

Output structure:
    data/pc_data/<dataset_name>/val/00000.npz, 00001.npz, ...
"""

import os
import sys
import argparse
import h5py
import trimesh
import numpy as np
from collections import OrderedDict

MAX_NUM_PART = 20
NUM_POINTS = 1000  # PuzzleFusion++ uses 1000 points per part


def load_meshes_from_hdf5(hdf5_file, item_path):
    """Load trimesh objects from an HDF5 item.

    Args:
        hdf5_file: open h5py File
        item_path: e.g. 'pig/synthetic_fracture/20/fractured_15'

    Returns:
        list of trimesh.Trimesh objects, one per piece
    """
    pieces_group = hdf5_file[item_path + "/pieces"]
    # Sort piece indices numerically
    piece_keys = sorted(pieces_group.keys(), key=lambda x: int(x))

    meshes = []
    for pk in piece_keys:
        verts = pieces_group[pk]["vertices"][()]
        faces = pieces_group[pk]["faces"][()]
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        meshes.append(mesh)

    return meshes


def compute_connectivity(meshes, max_num_part=MAX_NUM_PART, precision=5):
    """Compute boolean adjacency matrix by checking shared vertices.

    Matches PuzzleFusion++'s _check_connectivity / _are_meshes_connected logic.
    """
    num_parts = len(meshes)
    graph = np.zeros((max_num_part, max_num_part), dtype=bool)

    vertex_sets = []
    for mesh in meshes:
        rounded = np.round(mesh.vertices, decimals=precision)
        vertex_sets.append(set(map(tuple, rounded)))

    for i in range(num_parts):
        for j in range(i + 1, num_parts):
            shared = vertex_sets[i].intersection(vertex_sets[j])
            if len(shared) > 0:
                graph[i, j] = True
                graph[j, i] = True

    return graph


def sample_point_clouds(meshes, num_points=NUM_POINTS):
    """Sample point clouds from mesh surfaces.

    For very small meshes with fewer faces than requested points,
    fall back to vertex sampling with replacement.
    """
    pcs = []
    for mesh in meshes:
        if mesh.area > 0 and len(mesh.faces) > 0:
            pts, _ = trimesh.sample.sample_surface(mesh, num_points)
        else:
            # Degenerate mesh — sample vertices with replacement
            indices = np.random.choice(len(mesh.vertices), num_points, replace=True)
            pts = mesh.vertices[indices]
        pcs.append(pts)
    return np.stack(pcs, axis=0)  # [P, num_points, 3]


def select_ref_part(pcs, num_parts, max_num_part=MAX_NUM_PART):
    """Select reference part as the largest fragment by bounding box scale.

    Matches PuzzleFusion++'s GeometryPartDataset logic.
    """
    ref_part = np.zeros(max_num_part, dtype=bool)
    scales = np.max(pcs[:num_parts], axis=(1, 2)) - np.min(pcs[:num_parts], axis=(1, 2))
    ref_idx = np.argmax(scales)
    ref_part[ref_idx] = True
    return ref_part


def convert_dataset(
    hdf5_path,
    category,
    output_dir,
    max_parts=20,
    min_parts=2,
    split="val",
):
    """Convert one category from an HDF5 file to PuzzleFusion++ .npz files.

    Args:
        hdf5_path: path to the HDF5 file
        category: category name in the HDF5 (e.g. 'pig', 'ceramics')
        output_dir: output directory for .npz files
        max_parts: maximum number of parts to include (PuzzleFusion limit = 20)
        min_parts: minimum number of parts
        split: which split to use ('val' or 'train')
    """
    os.makedirs(output_dir, exist_ok=True)

    f = h5py.File(hdf5_path, "r")

    # Get item list from data_split
    split_key = f"data_split/{category}/{split}"
    if split_key not in f:
        # Try 'test' split as fallback
        split_key = f"data_split/{category}/test"
        if split_key not in f:
            print(f"  WARNING: No split found for {category}, listing all items")
            # Manually enumerate items
            items = []
            cat_group = f[category]

            def find_items(group, prefix):
                """Recursively find items that have a 'pieces' subgroup."""
                for key in group.keys():
                    path = f"{prefix}/{key}" if prefix else key
                    child = group[key]
                    if hasattr(child, "keys"):
                        if "pieces" in child:
                            items.append(f"{category}/{path}")
                        else:
                            find_items(child, path)

            find_items(cat_group, "")
            print(f"  Found {len(items)} items by enumeration")
        else:
            raw = f[split_key][()].tolist()
            items = [x.decode() if isinstance(x, bytes) else x for x in raw]
    else:
        raw = f[split_key][()].tolist()
        items = [x.decode() if isinstance(x, bytes) else x for x in raw]

    print(f"  Category '{category}': {len(items)} items in '{split}' split")

    saved = 0
    skipped_parts = 0
    skipped_mesh = 0

    for idx, item_path in enumerate(items):
        # Load meshes
        full_path = item_path + "/pieces"
        if full_path not in f:
            print(f"    SKIP {item_path}: no pieces group")
            skipped_mesh += 1
            continue

        pieces_group = f[full_path]
        num_parts = len(pieces_group.keys())

        # Filter by part count
        if num_parts < min_parts or num_parts > min(max_parts, MAX_NUM_PART):
            print(
                f"    SKIP {item_path}: {num_parts} parts "
                f"(limit {min_parts}-{min(max_parts, MAX_NUM_PART)})"
            )
            skipped_parts += 1
            continue

        # Load meshes and sample point clouds
        try:
            meshes = load_meshes_from_hdf5(f, item_path)
            pcs = sample_point_clouds(meshes, NUM_POINTS)
        except Exception as e:
            print(f"    SKIP {item_path}: mesh/sampling error: {e}")
            skipped_mesh += 1
            continue

        # Compute connectivity graph
        graph = compute_connectivity(meshes, MAX_NUM_PART)

        # Select reference part (largest by bounding box)
        ref_part = select_ref_part(pcs, num_parts, MAX_NUM_PART)

        # Build validity mask
        part_valids = np.zeros(MAX_NUM_PART, dtype=np.float32)
        part_valids[:num_parts] = 1.0

        # Save .npz
        out_path = os.path.join(output_dir, f"{saved:05d}.npz")
        np.savez(
            out_path,
            data_id=np.int64(saved),
            part_valids=part_valids,
            num_parts=np.int64(num_parts),
            mesh_file_path=item_path,
            graph=graph,
            category=category,
            part_pcs_gt=pcs,  # [P, 1000, 3]
            ref_part=ref_part,
        )
        saved += 1
        print(f"    [{saved}] {item_path}: {num_parts} parts -> {out_path}")

    f.close()

    print(
        f"  Done: {saved} saved, {skipped_parts} skipped (part count), "
        f"{skipped_mesh} skipped (mesh errors)"
    )
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hdf5",
        type=str,
        default=None,
        help="Path to one GARF-format .hdf5 (enables single-dataset mode).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for .npz files (required with --hdf5).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="artifact",
        help="data_split/<category>/ split to read (default: artifact).",
    )
    parser.add_argument("--min-parts", type=int, default=2)
    parser.add_argument("--max-parts", type=int, default=20)
    parser.add_argument("--split", type=str, default="val", choices=("val", "train", "test"))
    args = parser.parse_args()

    if args.hdf5:
        if not args.output_dir:
            parser.error("--output-dir is required when using --hdf5")
        print("=" * 60)
        print("HDF5 -> PuzzleFusion++ .npz (single dataset)")
        print("=" * 60)
        convert_dataset(
            hdf5_path=args.hdf5,
            category=args.category,
            output_dir=args.output_dir,
            max_parts=args.max_parts,
            min_parts=args.min_parts,
            split=args.split,
        )
        print("Done.")
        return

    base_out = "/data/gpfs/projects/punim2657/Puzzlefusion/data/pc_data"

    # ── Dataset definitions matching GARF eval_complex.slurm ──
    datasets = OrderedDict(
        [
            (
                "bone_syn_pig",
                {
                    "hdf5": "/data/gpfs/projects/punim2657/GARF/input/Fractura/bone_synthetic.hdf5",
                    "category": "pig",
                    "min_parts": 5,
                    "max_parts": 20,
                },
            ),
            (
                "bone_syn_rib",
                {
                    "hdf5": "/data/gpfs/projects/punim2657/GARF/input/Fractura/bone_synthetic.hdf5",
                    "category": "rib",
                    "min_parts": 5,
                    "max_parts": 20,  # PuzzleFusion limit; GARF used 50
                },
            ),
            (
                "fractura_real_ceramics",
                {
                    "hdf5": "/data/gpfs/projects/punim2657/GARF/input/Fractura/fractura_real.hdf5",
                    "category": "ceramics",
                    "min_parts": 2,
                    "max_parts": 20,
                },
            ),
            (
                "fractura_real_egg",
                {
                    "hdf5": "/data/gpfs/projects/punim2657/GARF/input/Fractura/fractura_real.hdf5",
                    "category": "egg",
                    "min_parts": 2,
                    "max_parts": 10,
                },
            ),
            (
                "fractura_real_bones",
                {
                    "hdf5": "/data/gpfs/projects/punim2657/GARF/input/Fractura/fractura_real.hdf5",
                    "category": "bones",
                    "min_parts": 2,
                    "max_parts": 10,
                },
            ),
        ]
    )

    print("=" * 60)
    print("HDF5 -> PuzzleFusion++ .npz Conversion")
    print("=" * 60)

    for name, cfg in datasets.items():
        output_dir = os.path.join(base_out, name, "val")
        print(f"\n── Converting: {name} ──")
        print(f"  HDF5: {cfg['hdf5']}")
        print(f"  Output: {output_dir}")
        convert_dataset(
            hdf5_path=cfg["hdf5"],
            category=cfg["category"],
            output_dir=output_dir,
            max_parts=cfg["max_parts"],
            min_parts=cfg["min_parts"],
            split="val",
        )

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
