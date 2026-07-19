#!/bin/bash
#SBATCH --job-name=pf_oracle_eval
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/data/gpfs/projects/punim2657/Puzzlefusion/logs/eval_oracle_%j.out
#SBATCH --error=/data/gpfs/projects/punim2657/Puzzlefusion/logs/eval_oracle_%j.err

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate puzzlefusionpp

cd /data/gpfs/projects/punim2657/Puzzlefusion
mkdir -p logs

export PYTHONPATH=$(pwd)

DENOISER_CKPT=output/denoiser/everyday_epoch2000_bs64/training/last.ckpt
VERIFIER_CKPT=output/verifier/everyday_epoch100_bs64/training/last.ckpt

echo "=== PuzzleFusion++ Complex Fragment Evaluation — FULL PIPELINE (oracle) ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Mode: denoise + verify + merge, max_iters=6"
echo "Matching data: oracle (synthetic from GT geometry)"
echo "Input data: assembly-normalized (pc_data_norm)"
echo ""

run_test() {
    local dataset=$1
    echo "══ $dataset ══"
    python test.py \
        experiment_name=everyday_epoch2000_bs64 \
        denoiser.data.val_batch_size=1 \
        denoiser.data.data_val_dir=./data/pc_data_norm/${dataset}/val/ \
        denoiser.data.matching_data_path=./data/matching_data_oracle/${dataset}/ \
        denoiser.ckpt_path=$DENOISER_CKPT \
        verifier.ckpt_path=$VERIFIER_CKPT \
        verifier.max_iters=6 \
        inference_dir=${dataset}_oracle
    echo "$dataset done at $(date)"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────
# Test 1: bone_syn_pig — 21 samples, 5-20 fragments
# ─────────────────────────────────────────────────────────────────────
run_test bone_syn_pig

# ─────────────────────────────────────────────────────────────────────
# Test 2: bone_syn_rib — 6 samples, 5-17 fragments (>20 excluded)
# ─────────────────────────────────────────────────────────────────────
run_test bone_syn_rib

# ─────────────────────────────────────────────────────────────────────
# Test 3: fractura_real_ceramics — 8 samples, 3-12 fragments
# ─────────────────────────────────────────────────────────────────────
run_test fractura_real_ceramics

# ─────────────────────────────────────────────────────────────────────
# Test 4: fractura_real_egg — 3 samples, 3-5 fragments
# ─────────────────────────────────────────────────────────────────────
run_test fractura_real_egg

# ─────────────────────────────────────────────────────────────────────
# Test 5: fractura_real_bones — 16 samples, 2-3 fragments
# ─────────────────────────────────────────────────────────────────────
run_test fractura_real_bones

echo "=== ALL 5 ORACLE EVALUATIONS COMPLETE ==="
echo "Date: $(date)"
echo ""
echo "Results saved under: output/denoiser/everyday_epoch2000_bs64/inference/<dataset>_oracle/"
