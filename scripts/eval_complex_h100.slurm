#!/bin/bash
#SBATCH --job-name=pf_complex_eval
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/data/gpfs/projects/punim2657/Puzzlefusion/logs/eval_complex_%j.out
#SBATCH --error=/data/gpfs/projects/punim2657/Puzzlefusion/logs/eval_complex_%j.err

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate puzzlefusionpp

cd /data/gpfs/projects/punim2657/Puzzlefusion
mkdir -p logs

export PYTHONPATH=$(pwd)

DENOISER_CKPT=output/denoiser/everyday_epoch2000_bs64/training/last.ckpt

echo "=== PuzzleFusion++ Complex Fragment Evaluation ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Mode: denoiser-only (no matching data for these datasets)"
echo ""

# ─────────────────────────────────────────────────────────────────────
# Test 1: bone_syn_pig — 21 samples, 5-20 fragments
# ─────────────────────────────────────────────────────────────────────
echo "══ Test 1: bone_syn_pig (5-20 frags, 21 samples) ══"
python test.py \
    experiment_name=everyday_epoch2000_bs64 \
    denoiser.data.val_batch_size=1 \
    denoiser.data.data_val_dir=./data/pc_data/bone_syn_pig/val/ \
    denoiser.ckpt_path=$DENOISER_CKPT \
    inference_dir=bone_syn_pig \
    verifier.max_iters=1

echo "Test 1 done at $(date)"
echo ""

# ─────────────────────────────────────────────────────────────────────
# Test 2: bone_syn_rib — 6 samples, 5-17 fragments (>20 excluded)
# ─────────────────────────────────────────────────────────────────────
echo "══ Test 2: bone_syn_rib (5-17 frags, 6 samples) ══"
python test.py \
    experiment_name=everyday_epoch2000_bs64 \
    denoiser.data.val_batch_size=1 \
    denoiser.data.data_val_dir=./data/pc_data/bone_syn_rib/val/ \
    denoiser.ckpt_path=$DENOISER_CKPT \
    inference_dir=bone_syn_rib \
    verifier.max_iters=1

echo "Test 2 done at $(date)"
echo ""

# ─────────────────────────────────────────────────────────────────────
# Test 3: fractura_real_ceramics — 8 samples, 3-12 fragments
# ─────────────────────────────────────────────────────────────────────
echo "══ Test 3: fractura_real_ceramics (3-12 frags, 8 samples) ══"
python test.py \
    experiment_name=everyday_epoch2000_bs64 \
    denoiser.data.val_batch_size=1 \
    denoiser.data.data_val_dir=./data/pc_data/fractura_real_ceramics/val/ \
    denoiser.ckpt_path=$DENOISER_CKPT \
    inference_dir=fractura_real_ceramics \
    verifier.max_iters=1

echo "Test 3 done at $(date)"
echo ""

# ─────────────────────────────────────────────────────────────────────
# Test 4: fractura_real_egg — 3 samples, 3-5 fragments
# ─────────────────────────────────────────────────────────────────────
echo "══ Test 4: fractura_real_egg (3-5 frags, 3 samples) ══"
python test.py \
    experiment_name=everyday_epoch2000_bs64 \
    denoiser.data.val_batch_size=1 \
    denoiser.data.data_val_dir=./data/pc_data/fractura_real_egg/val/ \
    denoiser.ckpt_path=$DENOISER_CKPT \
    inference_dir=fractura_real_egg \
    verifier.max_iters=1

echo "Test 4 done at $(date)"
echo ""

# ─────────────────────────────────────────────────────────────────────
# Test 5: fractura_real_bones — 16 samples, 2-3 fragments
# ─────────────────────────────────────────────────────────────────────
echo "══ Test 5: fractura_real_bones (2-3 frags, 16 samples) ══"
python test.py \
    experiment_name=everyday_epoch2000_bs64 \
    denoiser.data.val_batch_size=1 \
    denoiser.data.data_val_dir=./data/pc_data/fractura_real_bones/val/ \
    denoiser.ckpt_path=$DENOISER_CKPT \
    inference_dir=fractura_real_bones \
    verifier.max_iters=1

echo "Test 5 done at $(date)"
echo ""

echo "=== ALL 5 EVALUATIONS COMPLETE ==="
echo "Date: $(date)"
echo ""
echo "Results saved under: output/denoiser/everyday_epoch2000_bs64/inference/"
echo "  - bone_syn_pig/"
echo "  - bone_syn_rib/"
echo "  - fractura_real_ceramics/"
echo "  - fractura_real_egg/"
echo "  - fractura_real_bones/"
