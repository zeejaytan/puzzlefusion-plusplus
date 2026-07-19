#!/bin/bash
#SBATCH --job-name=pf-fix
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/gpfs/projects/punim2657/Puzzlefusion/fix_torch_%j.log

set -euo pipefail

PROJDIR=/data/gpfs/projects/punim2657/Puzzlefusion

echo "=== Fix started at $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate puzzlefusionpp

module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

echo ""
echo "=== Reinstalling torch 2.0.1+cu118 (overwrite the 2.4.1 that lightning pulled in) ==="
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 --force-reinstall --no-deps

echo ""
echo "=== Reinstalling triton 2.0.0 (torch 2.4.1 pulled in triton 3.0.0) ==="
pip install triton==2.0.0

echo ""
echo "=== Reinstalling torch-cluster for torch 2.0.1+cu118 ==="
pip install --force-reinstall torch-cluster -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

echo ""
echo "=== Rebuilding pytorch3d against correct torch ==="
cd "$PROJDIR/pytorch3d"
pip install -e . --no-build-isolation

echo ""
echo "=== Rebuilding chamferdist against correct torch ==="
cd "$PROJDIR/chamferdist"
python setup.py install

echo ""
echo "=== Verification ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    t = torch.zeros(1).cuda()
    print(f'CUDA tensor test: {t.device}')

import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
print('pytorch3d: OK (with CUDA ops)')

import torch_cluster
print('torch_cluster: OK')

import chamferdist
print('chamferdist: OK')

import hydra
print('hydra: OK')

import lightning
print('lightning: OK')

import diffusers
print('diffusers: OK')

import trimesh
print('trimesh: OK')

import wandb
print('wandb: OK')

import sklearn
print('scikit-learn: OK')

print()
print('=== ALL IMPORTS SUCCESSFUL ===')
"

echo ""
echo "=== Fix finished at $(date) ==="
