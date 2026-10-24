#!/bin/bash
#SBATCH --job-name=pf_env_imports
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/data/gpfs/projects/punim2657/PFpp/logs/env_imports_%j.out

# Proves the puzzlefusionpp environment still loads pytorch3d and chamferdist, and runs
# one CUDA op from each, on a compute node. Written for ticket 01 (builds moved out of
# the checkout to /data/gpfs/projects/punim2657/PFpp/build); rerun after any env change.
set -euo pipefail

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate puzzlefusionpp
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

cd /data/gpfs/projects/punim2657/PFpp
echo "=== $(hostname) $(date) ==="
python - <<'PY'
import torch, pytorch3d, chamferdist
from pytorch3d.ops import knn_points
from chamferdist import ChamferDistance
print("torch      ", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("pytorch3d  ", pytorch3d.__version__, pytorch3d.__file__)
print("chamferdist", chamferdist.__file__)
a = torch.rand(1, 500, 3, device="cuda"); b = torch.rand(1, 400, 3, device="cuda")
print("knn_points on GPU:", tuple(knn_points(a, b, K=4).dists.shape))
print("chamfer on GPU:   ", float(ChamferDistance()(a, b)))
import torch_cluster, hydra, lightning, diffusers
print("ALL IMPORTS OK")
PY
