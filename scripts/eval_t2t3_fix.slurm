#!/bin/bash
#SBATCH --job-name=pf_t2t3_fix
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/data/gpfs/projects/punim2657/Puzzlefusion/logs/eval_t2t3_fix_%j.out
#SBATCH --error=/data/gpfs/projects/punim2657/Puzzlefusion/logs/eval_t2t3_fix_%j.err

# Fix pass for T2 and T3-control (PFPP_JUGLET_SUCCESS plan):
#  - T2 rerun on SAME-BUILD geometry: the original t2 compared the May npz
#    (old hdf5 build) against a fresh dewear conversion (rebuilt hdf5) —
#    different frames/units, invalid pose deltas. Convert the current
#    juglet_deploy.hdf5 twice (resample control) + reuse the dewear npz.
#  - T3 control pairs: builder needs h5py -> run it with the GARF venv
#    (the conda env lacks h5py), then the 3 seed runs + scoring.
#  - T1c seam continuity on the original deploy layout (CPU).

set -euo pipefail
G=/data/gpfs/projects/punim2657/GARF
PF=/data/gpfs/projects/punim2657/Puzzlefusion
INF=$PF/output/denoiser/everyday_epoch2000_bs64/inference
cd $PF
mkdir -p logs
export PYTHONPATH=$PF

echo "### conversions + builders (GARF venv)"
$G/.venv/bin/python convert_hdf5_to_npz.py --hdf5 $G/input/juglet_deploy.hdf5 \
    --category artifact --output-dir $PF/data/pc_data/juglet_fresh_a/val
$G/.venv/bin/python convert_hdf5_to_npz.py --hdf5 $G/input/juglet_deploy.hdf5 \
    --category artifact --output-dir $PF/data/pc_data/juglet_fresh_b/val
$G/.venv/bin/python scripts/build_pair_npz.py control \
    --hdf5 $G/input/control_ceramics_pairs.hdf5 \
    --out $PF/data/pc_data/t3_control_pairs

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate puzzlefusionpp

echo "### T2 fresh runs (seed 123)"
for V in a b; do
    python test.py \
        experiment_name=everyday_epoch2000_bs64 \
        denoiser.data.val_batch_size=1 \
        denoiser.data.data_val_dir=./data/pc_data/juglet_fresh_${V}/val/ \
        denoiser.ckpt_path=output/denoiser/everyday_epoch2000_bs64/training/last.ckpt \
        inference_dir=t2_fresh_${V} \
        verifier.max_iters=1 \
        test_seed=123
done

echo "### T3 control pair runs"
for S in 41 42 43; do
    python test.py \
        experiment_name=everyday_epoch2000_bs64 \
        denoiser.data.val_batch_size=1 \
        denoiser.data.data_val_dir=./data/pc_data/t3_control_pairs/val/ \
        denoiser.ckpt_path=output/denoiser/everyday_epoch2000_bs64/training/last.ckpt \
        inference_dir=t3_control_s${S} \
        verifier.max_iters=1 \
        test_seed=${S}
done

echo "### T2b latent probes (center-matched)"
python scripts/vqvae_latent_probe.py \
    --npz-a data/pc_data/juglet_fresh_a/val/00000.npz \
    --npz-b data/pc_data/juglet_fresh_b/val/00000.npz \
    --ckpt output/denoiser/everyday_epoch2000_bs64/training/last.ckpt \
    --out logs/t2b_probe_resample_null
python scripts/vqvae_latent_probe.py \
    --npz-a data/pc_data/juglet_fresh_a/val/00000.npz \
    --npz-b data/pc_data/juglet_dewear/val/00000.npz \
    --ckpt output/denoiser/everyday_epoch2000_bs64/training/last.ckpt \
    --out logs/t2b_probe_wear

echo "### scoring (GARF venv)"
cd $G
TS=$(date +%Y%m%d_%H%M%S)
OUT=logs/diagnostics/pfpp_t2fix_${TS}
mkdir -p $OUT

$G/.venv/bin/python scripts/pfpp_pose_delta.py \
    --run-a "$INF/t2_fresh_a/0" --npz-a "$PF/data/pc_data/juglet_fresh_a/val/00000.npz" \
    --run-b "$INF/t2_fresh_b/0" --npz-b "$PF/data/pc_data/juglet_fresh_b/val/00000.npz" \
    --out $OUT/pose_delta_resample_null
$G/.venv/bin/python scripts/pfpp_pose_delta.py \
    --run-a "$INF/t2_fresh_a/0" --npz-a "$PF/data/pc_data/juglet_fresh_a/val/00000.npz" \
    --run-b "$INF/t2_dewear/0" --npz-b "$PF/data/pc_data/juglet_dewear/val/00000.npz" \
    --out $OUT/pose_delta_wear
$G/.venv/bin/python scripts/pfpp_layout_probes.py \
    --pfpp-dir "$INF/t2_fresh_a/0" \
    --pc-npz "$PF/data/pc_data/juglet_fresh_a/val/00000.npz" \
    --mesh-hdf5 $G/input/juglet_deploy.hdf5 --mesh-key artifact/Juglet-000 \
    --out $OUT --tag t2_fresh_a --n-null 0

echo "### T3 control scoring"
$G/.venv/bin/python scripts/pfpp_pair_chamfer.py control \
    --pairs-data "$PF/data/pc_data/t3_control_pairs" \
    --inference-dirs $INF/t3_control_s41 $INF/t3_control_s42 $INF/t3_control_s43 \
    --out logs/diagnostics/pfpp_t3_control_${TS}

echo "### T1c seam continuity (original deploy layout)"
$G/.venv/bin/python scripts/pfpp_seam_continuity.py \
    --pfpp-dir "$INF/juglet_deploy/0" \
    --pc-npz "$PF/data/pc_data/juglet_deploy/val/00000.npz" \
    --out logs/diagnostics/pfpp_t1c_${TS}

cp $PF/logs/t2b_probe_resample_null/latent_probe.json $OUT/latent_probe_resample_null.json || true
cp $PF/logs/t2b_probe_wear/latent_probe.json $OUT/latent_probe_wear.json || true
echo "=== DONE: $OUT ==="
