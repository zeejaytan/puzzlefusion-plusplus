#!/bin/bash
#SBATCH --job-name=pf_bones_retry
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/data/gpfs/projects/punim2657/Puzzlefusion/logs/eval_bones_retry_%j.out
#SBATCH --error=/data/gpfs/projects/punim2657/Puzzlefusion/logs/eval_bones_retry_%j.err

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate puzzlefusionpp

cd /data/gpfs/projects/punim2657/Puzzlefusion
export PYTHONPATH=$(pwd)

DENOISER_CKPT=output/denoiser/everyday_epoch2000_bs64/training/last.ckpt
VERIFIER_CKPT=output/verifier/everyday_epoch100_bs64/training/last.ckpt

echo "=== PF++ bones retry (fixed correspondence format) ==="
echo "Date: $(date)"

echo "══ fractura_real_bones ══"
python test.py \
    experiment_name=everyday_epoch2000_bs64 \
    denoiser.data.val_batch_size=1 \
    denoiser.data.data_val_dir=./data/pc_data_norm/fractura_real_bones/val/ \
    denoiser.data.matching_data_path=./data/matching_data_oracle/fractura_real_bones/ \
    denoiser.ckpt_path=$DENOISER_CKPT \
    verifier.ckpt_path=$VERIFIER_CKPT \
    verifier.max_iters=6 \
    inference_dir=fractura_real_bones_oracle

echo "fractura_real_bones done at $(date)"
