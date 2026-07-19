# PuzzleFusion++ Setup Log — 9 March 2026

## Step 1: Conda Environment Created

**Status:** Complete

Created a conda environment `puzzlefusionpp` with Python 3.8 on the Spartan HPC cluster.

### Commands Run

```bash
module load Anaconda3/2024.02-1
conda create --name puzzlefusionpp python=3.8 -y
```

### Result

- **Environment location:** `/home/zhuojiat/.conda/envs/puzzlefusionpp`
- **Python version:** 3.8.20
- **Python path:** `/home/zhuojiat/.conda/envs/puzzlefusionpp/bin/python`
- **Installed base packages:** pip 24.2, setuptools 75.1.0, wheel 0.44.0, sqlite 3.51.1, openssl 3.5.5

### Activation

To activate in future sessions:

```bash
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate puzzlefusionpp
```

Note: `conda activate` requires the shell hook (`eval "$(conda shell.bash hook)"`) since `conda init` has not been run on this system.

---

## Step 2: Install PyTorch 2.0.1+cu118

**Status:** Complete

Installed via pip from the cu118 index on a GPU node (A100 80GB, `gpu-a100-short` partition).

```bash
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

### Result
- **torch:** 2.0.1+cu118
- **torchvision:** 0.15.2+cu118
- **torchaudio:** 2.0.2+cu118
- **triton:** 2.0.0

---

## Step 3: Build pytorch3d, torch-cluster, chamferdist

**Status:** Complete

All three packages built/installed on GPU node with CUDA 11.8.

```bash
# pytorch3d (editable install from source)
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .

# torch-cluster (prebuilt wheel)
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# chamferdist (built from source with CUDA)
git clone https://github.com/krrish94/chamferdist.git
cd chamferdist && python setup.py install
```

### Result
- **pytorch3d:** 0.7.9 (editable, at `/data/gpfs/projects/punim2657/Puzzlefusion/pytorch3d`)
- **torch-cluster:** 1.6.3+pt20cu118
- **chamferdist:** 1.0.3

---

## Step 4: Install remaining pip packages

**Status:** Complete

```bash
pip install -r requirements.txt
pip install networkx scipy
pip install "huggingface_hub<0.24"  # fix: diffusers 0.21.4 needs cached_download API
```

### Issue encountered & fixed
`lightning==2.2.2` pulled in `torch==2.4.1` as a transitive dependency, overwriting `torch==2.0.1+cu118`. Fixed by force-reinstalling `torch==2.0.1+cu118` (with `--force-reinstall --no-deps`) and `triton==2.0.0` after the requirements install.

Also, `huggingface_hub==0.36.2` (pulled in by diffusers) removed the `cached_download` API that `diffusers==0.21.4` needs. Fixed by pinning `huggingface_hub<0.24`.

### Key installed packages
- hydra-core 1.3.2, lightning 2.2.2, diffusers 0.21.4
- imageio 2.31.6, wandb 0.15.12, scikit-learn 1.3.2, trimesh 4.0.2
- networkx 3.1, scipy 1.10.1, huggingface_hub 0.23.5

---

## Step 6: Verify all imports

**Status:** Complete

All imports verified on GPU node (SLURM job 22470405) and login node:
- torch 2.0.1+cu118, CUDA available, A100 80GB detected
- pytorch3d, torch_cluster, chamferdist — all OK
- hydra, lightning, diffusers, trimesh, wandb, scikit-learn, imageio, networkx, scipy — all OK

---

## Remaining Steps

- [ ] Step 5: Download data and pre-trained checkpoints from Google Drive
