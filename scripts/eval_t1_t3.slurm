#!/bin/bash
#SBATCH --job-name=pf_t1_t3
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/data/gpfs/projects/punim2657/Puzzlefusion/logs/eval_t1_t3_%j.out
#SBATCH --error=/data/gpfs/projects/punim2657/Puzzlefusion/logs/eval_t1_t3_%j.err

# T1a (sherd-identity scrambles), T1b (seed consistency), T3 (pairwise oracle)
# — PFPP_JUGLET_SUCCESS_EXPERIMENT_PLAN.md. All denoiser-only.

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate puzzlefusionpp

cd /data/gpfs/projects/punim2657/Puzzlefusion
mkdir -p logs
export PYTHONPATH=$(pwd)

DENOISER_CKPT=output/denoiser/everyday_epoch2000_bs64/training/last.ckpt
GARF=/data/gpfs/projects/punim2657/GARF

run () {  # run <data_val_dir> <inference_dir> <test_seed>
    echo "=== PF++ run: dir=$1 out=$2 seed=$3 ==="
    python test.py \
        experiment_name=everyday_epoch2000_bs64 \
        denoiser.data.val_batch_size=1 \
        denoiser.data.data_val_dir=$1 \
        denoiser.ckpt_path=$DENOISER_CKPT \
        inference_dir=$2 \
        verifier.max_iters=1 \
        test_seed=$3
}

echo "### build T1a scramble sets"
python scripts/build_t1a_scrambles.py \
    --base data/pc_data/juglet_deploy/val/00000.npz \
    --foreign-dir data/pc_data/fractura_real_ceramics/val \
    --out-root data/pc_data

echo "### build T3 pair sets"
python scripts/build_pair_npz.py juglet \
    --base data/pc_data/juglet_deploy/val/00000.npz \
    --out data/pc_data/t3_juglet_pairs
python scripts/build_pair_npz.py control \
    --hdf5 $GARF/input/control_ceramics_pairs.hdf5 \
    --out data/pc_data/t3_control_pairs

echo "### T1a arms (seed 123, same as deploy)"
for ARM in foreign dup mirror; do
    run ./data/pc_data/t1a_${ARM}/val/ t1a_${ARM} 123
done

echo "### T1b seed sweep on unmodified juglet"
for S in 41 42 43 44 45; do
    run ./data/pc_data/juglet_deploy/val/ t1b_seed${S} ${S}
done

echo "### T3 juglet pairs x3 seeds"
for S in 41 42 43; do
    run ./data/pc_data/t3_juglet_pairs/val/ t3_juglet_s${S} ${S}
done

echo "### T3 control pairs x3 seeds"
for S in 41 42 43; do
    run ./data/pc_data/t3_control_pairs/val/ t3_control_s${S} ${S}
done

echo "=== DONE ==="
