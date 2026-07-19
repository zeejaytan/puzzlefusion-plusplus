#!/bin/bash
#SBATCH --job-name=pf-inference-test
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/gpfs/projects/punim2657/Puzzlefusion/inference_test_%j.log

set -euo pipefail

PROJDIR=/data/gpfs/projects/punim2657/Puzzlefusion

echo "=== Job started at $(date) on $(hostname) ==="
nvidia-smi

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate puzzlefusionpp

module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

cd "$PROJDIR"

echo ""
echo "=== Quick import check ==="
python -c "
import torch
print(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
import pytorch3d, torch_cluster, chamferdist, hydra, lightning, diffusers
print('pytorch3d, torch_cluster, chamferdist, hydra, lightning, diffusers — all OK')
"

echo ""
echo "=== Running inference (100 val samples, 6 auto-aggl iters) ==="
python test.py \
    experiment_name=everyday_epoch2000_bs64 \
    denoiser.data.val_batch_size=1 \
    denoiser.data.data_val_dir=./data/pc_data/everyday/val/ \
    denoiser.data.matching_data_path=./data/matching_data/ \
    denoiser.data.overfit=100 \
    denoiser.ckpt_path=output/denoiser/everyday_epoch2000_bs64/training/last.ckpt \
    verifier.ckpt_path=output/verifier/everyday_epoch100_bs64/training/last.ckpt \
    inference_dir=test_run \
    verifier.max_iters=6

echo ""
echo "=== Job finished at $(date) ==="
