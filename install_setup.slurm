#!/bin/bash
#SBATCH --job-name=pf-setup
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data/gpfs/projects/punim2657/Puzzlefusion/install_setup_%j.log

set -euo pipefail

PROJDIR=/data/gpfs/projects/punim2657/Puzzlefusion

echo "=== Job started at $(date) on $(hostname) ==="
nvidia-smi

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate puzzlefusionpp

module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

echo ""
echo "=== Step 2: Installing PyTorch 2.0.1+cu118 ==="
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "=== Step 3a: Building pytorch3d from source ==="
cd "$PROJDIR"
if [ ! -d pytorch3d ]; then
    git clone https://github.com/facebookresearch/pytorch3d.git
fi
cd pytorch3d
pip install -e .

echo ""
echo "=== Step 3b: Installing torch-cluster ==="
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

echo ""
echo "=== Step 3c: Building chamferdist from source ==="
cd "$PROJDIR"
if [ ! -d chamferdist ]; then
    git clone https://github.com/krrish94/chamferdist.git
fi
cd chamferdist
python setup.py install

echo ""
echo "=== Step 4: Installing remaining pip packages ==="
cd "$PROJDIR"
pip install -r requirements.txt
pip install networkx scipy

echo ""
echo "=== Step 6: Verification ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
import pytorch3d
print(f'pytorch3d: OK')
import torch_cluster
print(f'torch_cluster: OK')
import chamferdist
print(f'chamferdist: OK')
import hydra
print(f'hydra: OK')
import lightning
print(f'lightning: OK')
import diffusers
print(f'diffusers: OK')
import trimesh
print(f'trimesh: OK')
print()
print('=== ALL IMPORTS SUCCESSFUL ===')
"

echo ""
echo "=== Job finished at $(date) ==="
