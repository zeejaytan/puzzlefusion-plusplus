#!/bin/bash
#SBATCH --job-name=puzzlefusion_pottery
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/data/gpfs/projects/punim2657/Puzzlefusion/logs/inference_pottery_%j.out
#SBATCH --error=/data/gpfs/projects/punim2657/Puzzlefusion/logs/inference_pottery_%j.err

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate puzzlefusionpp

cd /data/gpfs/projects/punim2657/Puzzlefusion
mkdir -p logs

export PYTHONPATH=$(pwd)

python test.py \
    experiment_name=everyday_epoch2000_bs64 \
    denoiser.data.val_batch_size=1 \
    denoiser.data.data_val_dir=./data/pc_data/everyday/val_pottery/ \
    denoiser.data.matching_data_path=./data/matching_data/ \
    denoiser.ckpt_path=output/denoiser/everyday_epoch2000_bs64/training/last.ckpt \
    verifier.ckpt_path=output/verifier/everyday_epoch100_bs64/training/last.ckpt \
    inference_dir=pottery_run \
    verifier.max_iters=6
