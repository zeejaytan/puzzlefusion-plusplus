#!/bin/bash
#SBATCH --job-name=pf_juglet_eval
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/data/gpfs/projects/punim2657/Puzzlefusion/logs/eval_juglet_%j.out
#SBATCH --error=/data/gpfs/projects/punim2657/Puzzlefusion/logs/eval_juglet_%j.err

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate puzzlefusionpp

cd /data/gpfs/projects/punim2657/Puzzlefusion
mkdir -p logs

export PYTHONPATH=$(pwd)

DENOISER_CKPT=output/denoiser/everyday_epoch2000_bs64/training/last.ckpt

echo "=== PuzzleFusion++ Juglet (9 parts, 1 sample) ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Mode: denoiser-only (verifier.max_iters=1)"
echo ""

python test.py \
    experiment_name=everyday_epoch2000_bs64 \
    denoiser.data.val_batch_size=1 \
    denoiser.data.data_val_dir=./data/pc_data/juglet_archaeological/val/ \
    denoiser.ckpt_path=$DENOISER_CKPT \
    inference_dir=juglet_archaeological \
    verifier.max_iters=1

echo ""
echo "=== DONE ==="
echo "Results: output/denoiser/everyday_epoch2000_bs64/inference/juglet_archaeological/"
echo "Date: $(date)"
