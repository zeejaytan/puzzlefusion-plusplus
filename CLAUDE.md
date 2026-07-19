# CLAUDE.md — PuzzleFusion++ (project)

Follow the workspace root **`../AGENTS.md`** / **`../CLAUDE.md`** (laptop ↔ GitHub ↔ Spartan) for all shared rules. Same overlay as **`AGENTS.md`** in this folder (fork/upstream/`REMOTE_ROOT` table lives there). Edit and commit on the laptop; Spartan is pull-only (`git pull --ff-only`) and runs Slurm via `scripts/remote/*`. Heavy data (`data/`, `output/`, `logs`, checkpoints, `*.hdf5`) stays on Spartan; `artifacts/` is the local, gitignored rsync landing zone.

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)

# PuzzleFusion++ Codebase Reference

**Paper:** [PuzzleFusion++: Auto-agglomerative 3D Fracture Assembly by Denoise and Verify](https://arxiv.org/abs/2406.00259) (ICLR 2025)
**Repo:** <https://github.com/zeejaytan/puzzlefusion-plusplus> (fork of [eric-zqwang/puzzlefusion-plusplus](https://github.com/eric-zqwang/puzzlefusion-plusplus))

---

## Overview

PuzzleFusion++ is a fully neural "auto-agglomerative" 3D fracture assembly system. Given a set of fractured 3D fragments as point clouds, it predicts each fragment's 6-DoF pose (translation + rotation) to reassemble the original object.

The key idea (from the paper, Section 3): starting from individual fragments, the system (1) simultaneously denoises the 6-DoF alignment parameters of all fragments using a diffusion model, (2) verifies and merges pairwise alignments using a transformer, and (3) repeats this process iteratively — resembling how humans solve jigsaw puzzles.

### Three Trained Modules

| Module | Paper Section | Code | Purpose |
|--------|--------------|------|---------|
| **VQVAE** (Fragment Autoencoder) | §3.1, Appendix A.1 | `puzzlefusion_plusplus/vqvae/` | Encodes each fragment's 1000-point cloud into 25 quantized local latent vectors (64D each) |
| **SE3 Denoiser** | §3.2 | `puzzlefusion_plusplus/denoiser/` | DDPM diffusion model that denoises 7D pose parameters (4D quaternion + 3D translation) for all fragments simultaneously |
| **Pairwise Alignment Verifier** | §3.3 | `puzzlefusion_plusplus/verifier/` | Transformer that classifies whether each pair of fragments is correctly aligned, enabling merge decisions |

### Auto-Agglomerative Inference (Paper §3.4)

The full inference loop lives in `puzzlefusion_plusplus/auto_aggl.py` (`AutoAgglomerative` class). It:

1. Runs the denoiser for 20 DDPM sampling steps
2. Runs the verifier on all fragment pairs to get alignment scores
3. Merges verified pairs (score > 0.9) into larger fragments using `networkx` graph operations
4. Removes inner-surface points from merged fragments and resamples via FPS
5. Repeats up to 6 iterations (configurable via `verifier.max_iters`)

---

## Directory Structure

```
Puzzlefusion/
├── train_vqvae.py              # Stage 1: Train the fragment autoencoder
├── train_denoiser.py           # Stage 2: Train the SE3 denoiser
├── train_verifier.py           # Stage 3: Train the pairwise verifier
├── test.py                     # Inference: auto-agglomerative assembly
├── generate_pc_data.py         # Preprocessing: mesh → point cloud .npz files
│
├── puzzlefusion_plusplus/
│   ├── auto_aggl.py            # AutoAgglomerative: full inference pipeline
│   ├── vqvae/                  # Fragment Autoencoder module
│   │   ├── model/
│   │   │   ├── fracture_ae.py          # FractureAE (LightningModule wrapper)
│   │   │   └── modules/
│   │   │       ├── pn2.py              # PointNet++ encoder + MLP decoder
│   │   │       ├── vq_vae.py           # VQVAE: encode → quantize → decode
│   │   │       └── quantizer.py        # VectorQuantizer (1024 embeddings, 16D each)
│   │   ├── dataset/
│   │   │   ├── dataset.py              # GeometryPartDataset: reads Breaking Bad meshes
│   │   │   └── pc_dataset.py           # Point cloud dataset variant
│   │   └── data/
│   │       └── data_module.py          # Lightning DataModule for VQVAE training
│   │
│   ├── denoiser/               # SE3 Denoiser module
│   │   ├── model/
│   │   │   ├── denoiser.py             # Denoiser (LightningModule): forward, loss, train/val
│   │   │   └── modules/
│   │   │       ├── denoiser_transformer.py  # DenoiserTransformer: the core architecture
│   │   │       ├── attention.py             # EncoderLayer: local SA + global SA + FFN
│   │   │       ├── custom_diffusers.py      # PiecewiseScheduler (custom DDPM scheduler)
│   │   │       └── encoder.py               # VQVAE wrapper used inside the denoiser
│   │   ├── dataset/
│   │   │   └── dataset.py              # GeometryLatentDataset: reads preprocessed .npz
│   │   └── evaluation/
│   │       ├── evaluator.py            # Part Accuracy, Shape CD, RMSE(R), RMSE(T)
│   │       └── transform.py            # Quaternion/transform utilities
│   │
│   └── verifier/               # Pairwise Alignment Verifier module
│       ├── model/
│       │   ├── verifier.py             # Verifier (LightningModule): BCE loss, metrics
│       │   └── modules/
│       │       └── verifier_transformer.py  # VerifierTransformer: standard Transformer
│       └── dataset/
│           └── dataset.py              # VerifierDataset: edge features + labels
│
├── config/                     # Hydra YAML configs
│   ├── ae/                     # VQVAE configs (model, data, vq_vae, global_config)
│   ├── denoiser/               # Denoiser configs (model, data, encoder, global_config)
│   ├── verifier/               # Verifier configs (model, global_config)
│   └── auto_aggl.yaml          # Full inference config (combines all three)
│
├── utils/
│   ├── model_utils.py          # PositionalEncoding, EmbedderNerf (NeRF-style freq encoding)
│   ├── node_merge_utils.py     # Graph merge, FPS downsampling, inner-point removal
│   └── pn2_utils.py            # PointNet++ set abstraction layers
│
├── Jigsaw_matching/            # Jigsaw point matcher (used for verifier data generation)
│   ├── model/                  # Jigsaw model (PointNet2, DGCNN, attention, affinity)
│   ├── dataset/                # Matching dataset
│   ├── utils/                  # Matching utilities (chamfer, transforms, alignment)
│   └── experiments/            # Jigsaw experiment configs
│
├── renderer/                   # Blender-based visualization
├── scripts/                    # Shell scripts for training/inference
├── docs/                       # Installation, data prep, training, test guides
└── requirements.txt            # hydra-core, lightning, diffusers, wandb, trimesh, etc.
```

---

## Architecture Details (Cross-referenced with Paper)

### 1. VQVAE — Fragment Autoencoder (Paper §3.1, Appendix A.1)

**Paper:** "We train PointNet++ and VQ-VAE to encode the point clouds into latent vectors at the points... PointNet++ employs furthest point sampling to select 25 points, creating 25 corresponding point latent vectors for each fragment."

**Code:** `puzzlefusion_plusplus/vqvae/model/modules/`

- **Encoder** (`pn2.py` → `PN2`): Three PointNet++ Set Abstraction layers:
  - SA1: 1000→256 points, radius=0.2, 32 neighbors, MLP [64, 64, 128]
  - SA2: 256→128 points, radius=0.4, 64 neighbors, MLP [128, 128, 256]
  - SA3: 128→**25 points**, radius=0.8, 64 neighbors, MLP [256, 256, 512]
  - Conv1d: 512 → **64** channels → produces `z_e ∈ R^{25×64}` (25 point latents, each 64D)

- **Vector Quantizer** (`quantizer.py` → `VectorQuantizer`):
  - Codebook: **1024 embeddings**, each **16D**
  - Each 64D latent is split into 4×16D, quantized independently, then concatenated back
  - Loss: `||sg[z_q] - z_e||² + β||z_q - sg[z_e]||²` with β=0.25

- **Decoder** (`pn2.py` → `PN2.decode`): MLP [64→256→512→40×3], reconstructs 40 local points per latent center
  - Reconstruction: `P_rec = D(z_p) + c_p` for each of 25 centers → union of 25×40 = 1000 points
  - Loss: Bidirectional Chamfer Distance vs. original 1000-point cloud

**Config:** `config/ae/` — trained for 2000 epochs, AdamW lr=5e-4

### 2. SE3 Denoiser (Paper §3.2)

**Paper:** "Each sampled point of each fragment keeps track of the alignment estimate in the denoise transformer architecture... 1) Intra-fragment self-attention (Local SA) among 25 sampled points within a fragment; 2) Inter-fragment self-attention (Global SA) among all sampled points across all fragments; and 3) Average pooling over the 25 sampled points of fragment f to predict the residual."

**Code:** `puzzlefusion_plusplus/denoiser/model/`

- **Denoiser** (`denoiser.py` → `Denoiser` LightningModule):
  - Training forward: sample random timestep per batch, add noise via PiecewiseScheduler, predict noise ε
  - Loss: MSE between predicted and true noise on non-anchor fragments
  - Validation: full 20-step DDPM sampling + metric evaluation

- **DenoiserTransformer** (`modules/denoiser_transformer.py`):
  - Feature embedding per point `x̂_{p,t} ∈ R^512` is formed by combining:
    1. **Alignment estimate** `x_t^f ∈ R^7` → NeRF positional encoding → `R^147` → Linear → `R^512`
    2. **Point latent** `z_q ∈ R^64` from frozen VQVAE encoder
    3. **Point coordinate** `c_p ∈ R^3` → NeRF positional encoding → `R^63`
    4. **Scale factor** `n^f ∈ R^1` → NeRF positional encoding → `R^21`
    - Items 2-4 concatenated → Linear → `R^512` (shape embedding)
    - Sum of alignment embedding + shape embedding + positional encoding

  - **6 EncoderLayers** (`attention.py`), each with:
    1. **Local SA (Intra-fragment)**: self-attention among 25 points within each fragment, using block-diagonal mask
    2. **Global SA (Inter-fragment)**: self-attention across all points of all valid fragments
    3. **FFN**: GEGLU feed-forward
    - Both SA layers use **Adaptive LayerNorm** (AdaLN) to inject timestep `t` via learned embedding

  - **Output heads**: Average pool 25 points → MLP [512→512→256→3] for translation, MLP [512→512→256→4] for rotation

  - **Anchor fragments**: Processed normally but alignment output is discarded; no gradient injected during training

  - **Architecture hyperparameters** (from `config/denoiser/model.yaml`):
    - embed_dim=512, num_layers=6, num_heads=8, dropout=0.1
    - multires=10 (NeRF positional encoding frequency bands)

- **Noise Scheduler** (`modules/custom_diffusers.py` → `PiecewiseScheduler`):
  - Extends `DDPMScheduler` with a **piecewise-quadratic** alpha schedule (Paper Appendix A.2, Fig 7):
    - t ∈ [0, 700]: `ᾱ(t) = 1 - 0.1·(t/700)²` (slow noise for coarse alignment)
    - t ∈ [700, 1000]: `ᾱ(t) = 0.9·(1 - ((t-700)/300)²)` (fast noise for fine alignment)
  - This allocates more denoising budget to fine alignment vs. coarse positioning, matching the paper's motivation that "accurately aligning local fracture surfaces is usually more challenging than knowing the rough location"

- **Training** (Paper §4): 2000 epochs, batch_size=64, AdamW lr=2e-4 (decay ×0.5 at epochs 1200, 1700), 4× RTX A6000

### 3. Pairwise Alignment Verifier (Paper §3.3)

**Paper:** "Given the alignment of F fragments, the verifier employs a Transformer architecture with C(F,2) nodes to perform binary classification on the correctness of C(F,2) pairwise alignments simultaneously."

**Code:** `puzzlefusion_plusplus/verifier/model/`

- **VerifierTransformer** (`modules/verifier_transformer.py`):
  - Input per edge (pair): 7D feature vector = 6D normalized histogram of Chamfer distances at bin thresholds [0, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, ∞] + total match count
  - Embedding: Linear(7 → 256) + positional encoding of fragment indices (PE for nodes 0–19)
  - Standard `TransformerEncoder`: 6 layers, 8 heads, dim=256, FFN dim=2048, GELU, dropout=0.1
  - Output: Linear(256 → 1) → sigmoid → binary classification

- **Verifier** (`verifier.py` → `Verifier` LightningModule):
  - Loss: BCE with class weighting (0.2 for negatives, 1.0 for positives)
  - Metrics: accuracy, precision, recall, F1

- **Training**: 100 epochs, AdamW lr=2e-4, single GPU

### 4. Auto-Agglomerative Inference (Paper §3.4)

**Code:** `puzzlefusion_plusplus/auto_aggl.py` → `AutoAgglomerative`

The `test_step` method implements the full paper algorithm:

```
For iter = 1 to max_iters (default 6):
  1. DENOISE: Run DDPM reverse process (20 steps) with frozen VQVAE + denoiser
     - At each step: extract features → predict noise → scheduler.step()
     - Anchor fragment poses reset to ground truth after each step

  2. VERIFY (if not last iteration):
     - For each edge in the Jigsaw matching data, compute CD histogram
     - Run verifier transformer → sigmoid scores
     - Edges with score > threshold (0.9) are classified as correct

  3. MERGE:
     - Non-anchor pairs with verified alignment are merged via networkx connected components
     - Merged fragments: combine point clouds → remove inner-surface points (normal-based heuristic using pytorch3d.ops.estimate_pointcloud_normals) → FPS downsample to 1000 points
     - Update graph: set pivot to largest fragment, recenter, update scales

  4. Stop early if all larger parts are classified
```

The denoiser-only mode (`max_iters=1`) skips verification and merging entirely.

---

## Data Pipeline

### Breaking Bad Dataset

The system uses the [Breaking Bad Dataset](https://breaking-bad-dataset.github.io/) (everyday subset: 34,075 train / 7,679 test assemblies from 407/91 objects, up to 20 fragments each).

### Preprocessing Steps

1. **Raw mesh → Point clouds** (`generate_pc_data.py`):
   - Reads mesh files from Breaking Bad, samples 1000 points per fragment
   - Builds connectivity graph (shared vertices between fragments)
   - Selects anchor = largest fragment (by AABB longest dimension)
   - Saves as `.npz` files to `data/pc_data/everyday/{train,val}/`

2. **Matching data** (via `Jigsaw_matching/`):
   - Jigsaw point matcher identifies corresponding surface points between fragment pairs
   - Saves edges, correspondences, critical surface points to `data/matching_data/`
   - Used at test time by the verifier

3. **Verifier training data** (separate generation):
   - Pre-computed edge features (CD histograms) + ground truth labels
   - Saved to `data/verifier_data/`

### Data Format at Training Time

For the **denoiser** (`puzzlefusion_plusplus/denoiser/dataset/dataset.py` → `GeometryLatentDataset`):

Each sample is prepared by:
1. Apply random global rotation to the whole assembly
2. Recenter so anchor fragment's centroid is at origin
3. For each fragment: recenter → apply random rotation → record GT translation + quaternion
4. Normalize each fragment's point cloud to [-1, 1] → record scale factor
5. Pad to `max_num_part=20` with zeros

The `data_dict` contains:
- `part_pcs`: [P, 1000, 3] — normalized fragment point clouds
- `part_trans`: [P, 3] — GT translations
- `part_rots`: [P, 4] — GT quaternions (scalar-first: [w, x, y, z])
- `part_valids`: [P] — 1 for real parts, 0 for padding
- `ref_part`: [P] — boolean mask for anchor fragment(s)
- `part_scale`: [P, 1] — normalization scale per fragment

**Multiple reference parts** (Paper §3.1): During training, with 50% probability, neighboring fragments of the anchor are also set as references with slightly noisy poses (timestep < 50) to simulate the merging process at inference time.

---

## Configuration System

All configs use [Hydra](https://hydra.cc/) (v1.3.2). Each module has its own `config/` subdirectory:

| Config | Entry point | Key settings |
|--------|-------------|-------------|
| `config/ae/global_config.yaml` | `train_vqvae.py` | 2000 epochs, monitor val_loss/cd_loss |
| `config/denoiser/global_config.yaml` | `train_denoiser.py` | 2000 epochs, val every 100, monitor eval/part_acc |
| `config/verifier/global_config.yaml` | `train_verifier.py` | 100 epochs, val every 5, monitor val/cls_acc |
| `config/auto_aggl.yaml` | `test.py` | Combines denoiser+verifier+ae, max_iters=6, threshold=0.9 |

### Key Model Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| DDPM timesteps | 1000 | `config/denoiser/model.yaml` |
| Inference steps | 20 | `config/denoiser/model.yaml` |
| Denoiser layers | 6 | `config/denoiser/model.yaml` |
| Denoiser embed_dim | 512 | `config/denoiser/model.yaml` |
| Denoiser heads | 8 | `config/denoiser/model.yaml` |
| Verifier layers | 6 | `config/verifier/model.yaml` |
| Verifier embed_dim | 256 | `config/verifier/model.yaml` |
| VQ codebook size | 1024 × 16D | `config/denoiser/encoder.yaml` |
| PN2 latent points | 25 | `config/denoiser/encoder.yaml` |
| PN2 latent dim | 64 | `config/denoiser/encoder.yaml` |
| Max fragments | 20 | `config/denoiser/data.yaml` |
| Points per fragment | 1000 | Dataset code |

---

## Evaluation Metrics (Paper §4)

Implemented in `puzzlefusion_plusplus/denoiser/evaluation/evaluator.py`:

| Metric | Code function | Description |
|--------|--------------|-------------|
| **RMSE(R)** | `rot_metrics()` | RMSE of Euler angle differences (degrees), with 360° wraparound handling |
| **RMSE(T)** | `trans_metrics()` | RMSE of translation differences |
| **Part Accuracy** | `calc_part_acc()` | % of fragments with bidirectional Chamfer Distance < 0.01 after alignment |
| **Shape CD** | `calc_shape_cd()` | Per-assembly Chamfer Distance of the full reconstructed shape |

### Expected Results (from `docs/test.md`)

Full auto-agglomerative (6 iterations):
- Part Accuracy: **70.18%**
- RMSE(R): **38.47°**
- RMSE(T): **0.0797**
- Shape CD: **0.00657**

---

## Training Pipeline

### Stage 1: VQVAE (`train_vqvae.py`)

```bash
sh ./scripts/train_vqvae.sh
```

- Trains PointNet++ encoder + VQ codebook + MLP decoder
- Loss: Bidirectional Chamfer Distance + VQ embedding loss
- Input: raw mesh → sample 1000 points per fragment
- Output checkpoint: `output/autoencoder/{experiment_name}/`

### Stage 2: SE3 Denoiser (`train_denoiser.py`)

```bash
sh ./scripts/train_denoiser.sh
```

- Loads frozen VQVAE encoder weights (set `model.encoder_weights_path` in config)
- Trains DenoiserTransformer with DDPM noise prediction loss
- Input: preprocessed `.npz` point cloud data (from `generate_pc_data.py`)
- Output checkpoint: `output/denoiser/{experiment_name}/`

### Stage 3: Verifier (`train_verifier.py`)

```bash
sh ./scripts/train_verifier.sh
```

- Trains on pre-generated verifier data (edge features + binary labels from Jigsaw)
- Loss: weighted BCE (0.2 for negatives, 1.0 for positives)
- Output checkpoint: `output/verifier/{experiment_name}/`

### Inference (`test.py`)

```bash
sh ./scripts/inference.sh
```

- Loads denoiser + VQVAE encoder + verifier checkpoints
- Runs `AutoAgglomerative.test_step()` on validation set (batch_size=1)
- Saves per-sample predictions to `output/denoiser/{experiment_name}/inference/{inference_dir}/`

---

## Key Implementation Details

### Quaternion Convention

The codebase uses **scalar-first** quaternions [w, x, y, z]. When interfacing with `scipy.spatial.transform.Rotation` (which uses scalar-last [x, y, z, w]), indices are swapped:
```python
quat_gt = quat_gt[[3, 0, 1, 2]]  # scipy → codebase
```

### Anchor Fragment Logic

- Anchor = fragment with largest AABB extent (`np.argmax(scale)`)
- Anchor's pose is fixed to identity rotation + zero translation
- During training: optionally expand anchors to neighbors with noisy poses (50% chance)
- During inference: any fragment merged with anchor also becomes an anchor

### Inner Surface Point Removal (Paper §3.4)

In `utils/node_merge_utils.py` → `remove_intersect_points_and_fps_ds()`:
- Estimate normals via `pytorch3d.ops.estimate_pointcloud_normals(neighborhood_size=20)`
- A point is "inner" if: another point in the opposing fragment is within distance 0.001 AND their normals have negative dot product
- After removal, FPS downsample back to 1000 points

### Feature Re-extraction

The VQVAE encoder re-encodes fragments at every denoising step because the encoding is rotation-sensitive. The current noisy rotation is applied to the point cloud before encoding:
```python
part_pcs = self._apply_rots(part_pcs, noisy_trans_and_rots)
encoder_out = self.encoder.encode(part_pcs)
```

---

## Dependencies

From `requirements.txt` and `docs/installation.md`:

- Python 3.8
- PyTorch 2.0.1 (CUDA 11.8)
- pytorch3d (from source)
- torch-cluster (for FPS in merge step)
- chamferdist (Chamfer Distance computation)
- hydra-core 1.3.2
- lightning 2.2.2
- diffusers 0.21.4 (for DDPMScheduler base class)
- wandb 0.15.12
- trimesh 4.0.2
- networkx (for connected components in auto-aggl)
- scipy (for Rotation)

---

## Comparison with GARF

Both PuzzleFusion++ and GARF (located at `/data/gpfs/projects/punim2657/GARF/`) tackle 3D fracture reassembly on the Breaking Bad dataset, but differ substantially:

| Aspect | PuzzleFusion++ | GARF |
|--------|---------------|------|
| **Backbone** | PointNet++ + VQVAE (25 latent points, 64D) | PointTransformerV3 (dense point features) |
| **Noise model** | DDPM with piecewise-quadratic schedule | Flow matching (or DDPM variant) |
| **Denoiser output** | Predicts noise ε (standard DDPM) | Predicts velocity field (flow matching) |
| **Iterative refinement** | Auto-agglomerative: denoise → verify → merge → repeat (up to 6 iters) | Multi-iteration re-denoising (no verify/merge) with optional one-step init |
| **Verifier** | Separate transformer for pairwise alignment verification | None |
| **Pre-training** | VQVAE autoencoder (self-supervised) | Fracture-aware segmentation (coarse seg head + Dice loss) |
| **Data format** | Pre-processed .npz (mesh → 1000 pts offline) | HDF5 (online loading) |
| **Fine-tuning** | Not supported | LoRA-based fine-tuning on custom datasets |
| **Config system** | Hydra (separate configs per module) | Hydra (unified experiment overrides) |
| **Quaternion handling** | Scalar-first [w,x,y,z], standard Gaussian noise | Scalar-first, uses scipy random rotations for noise |
| **Feature encoding** | VQ-quantized local features (frozen at denoiser training) | Frozen PTv3 backbone features |
| **Training scale** | Everyday subset only (34K assemblies) | Up to everyday+artifact+other (1.9M fractures) |
| **Reported PA (everyday)** | 70.6% | 94.77% (GARF-mini) / 95.68% (GARF) |

---

## Quick Start Commands

```bash
cd /data/gpfs/projects/punim2657/Puzzlefusion

# 1. Install dependencies (see docs/installation.md for full setup)
pip install -r requirements.txt

# 2. Preprocess data (if you have Breaking Bad meshes)
python generate_pc_data.py +data.save_pc_data_path=data/pc_data/everyday/

# 3. Train VQVAE
python train_vqvae.py experiment_name=vqvae_run1

# 4. Train Denoiser (set encoder weights path)
python train_denoiser.py experiment_name=denoiser_run1 \
  model.encoder_weights_path=output/autoencoder/vqvae_run1/training/last.ckpt

# 5. Train Verifier
python train_verifier.py experiment_name=verifier_run1

# 6. Inference
python test.py \
  denoiser.ckpt_path=output/denoiser/denoiser_run1/training/last.ckpt \
  verifier.ckpt_path=output/verifier/verifier_run1/last.ckpt \
  experiment_name=denoiser_run1 \
  inference_dir=results
```
