#!/bin/bash
#SBATCH --job-name=pf_t4_ablation
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/data/gpfs/projects/punim2657/Puzzlefusion/logs/eval_t4_ablations_%j.out
#SBATCH --error=/data/gpfs/projects/punim2657/Puzzlefusion/logs/eval_t4_ablations_%j.err

# T4 — prior-dominance ablation (PFPP_JUGLET_SUCCESS_EXPERIMENT_PLAN.md).
# Corrupt PF++'s shape conditioning on the Juglet deploy sample while keeping
# scale + anchor intact:
#   shuffle — permute (z_q, xyz) between sherds (identities swapped)
#   random  — random codebook tokens (content garbled, support kept)
#   zero    — zeroed latents
# Plus a "none" control re-run with the same seed for a like-for-like baseline.

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate puzzlefusionpp

cd /data/gpfs/projects/punim2657/Puzzlefusion
mkdir -p logs
export PYTHONPATH=$(pwd)

DENOISER_CKPT=output/denoiser/everyday_epoch2000_bs64/training/last.ckpt

for MODE in none shuffle random zero; do
    echo "=== T4 ablation_mode=$MODE ==="
    python test.py \
        experiment_name=everyday_epoch2000_bs64 \
        denoiser.data.val_batch_size=1 \
        denoiser.data.data_val_dir=./data/pc_data/juglet_deploy/val/ \
        denoiser.ckpt_path=$DENOISER_CKPT \
        inference_dir=t4_${MODE} \
        verifier.max_iters=1 \
        test_seed=123 \
        +ablation_mode=${MODE}
done

echo "=== DONE: output/denoiser/everyday_epoch2000_bs64/inference/t4_{none,shuffle,random,zero}/ ==="
