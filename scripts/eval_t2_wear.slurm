#!/bin/bash
#SBATCH --job-name=pf_t2_wear
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/data/gpfs/projects/punim2657/Puzzlefusion/logs/eval_t2_wear_%j.out
#SBATCH --error=/data/gpfs/projects/punim2657/Puzzlefusion/logs/eval_t2_wear_%j.err

# T2 — wear-invariance of the PF++ channel (PFPP_JUGLET_SUCCESS plan).
# T2a: run PF++ on the exp14 de-weathered juglet (same seed as the original
#      deploy) and measure per-part pose deltas + layout panel.
# T2b: VQVAE latent probe original vs de-weathered (mirror of GARF Exp 10).

set -uo pipefail
G=/data/gpfs/projects/punim2657/GARF
PF=/data/gpfs/projects/punim2657/Puzzlefusion
INF=$PF/output/denoiser/everyday_epoch2000_bs64/inference
cd $PF
mkdir -p logs
export PYTHONPATH=$PF

echo "### convert de-weathered hdf5 -> npz (GARF venv: h5py+trimesh)"
$G/.venv/bin/python convert_hdf5_to_npz.py \
    --hdf5 $G/input/juglet_deploy_dewear.hdf5 \
    --category artifact \
    --output-dir $PF/data/pc_data/juglet_dewear/val

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate puzzlefusionpp

echo "### T2a run on de-weathered juglet (seed 123 = original deploy seed)"
python test.py \
    experiment_name=everyday_epoch2000_bs64 \
    denoiser.data.val_batch_size=1 \
    denoiser.data.data_val_dir=./data/pc_data/juglet_dewear/val/ \
    denoiser.ckpt_path=output/denoiser/everyday_epoch2000_bs64/training/last.ckpt \
    inference_dir=t2_dewear \
    verifier.max_iters=1 \
    test_seed=123

echo "### T2b latent probe"
python scripts/vqvae_latent_probe.py \
    --npz-a data/pc_data/juglet_deploy/val/00000.npz \
    --npz-b data/pc_data/juglet_dewear/val/00000.npz \
    --ckpt output/denoiser/everyday_epoch2000_bs64/training/last.ckpt \
    --out logs/t2b_latent_probe

echo "### T2a scoring (GARF venv)"
cd $G
TS=$(date +%Y%m%d_%H%M%S)
$G/.venv/bin/python scripts/pfpp_pose_delta.py \
    --run-a "$INF/t4_none/0" --npz-a "$PF/data/pc_data/juglet_deploy/val/00000.npz" \
    --run-b "$INF/t2_dewear/0" --npz-b "$PF/data/pc_data/juglet_dewear/val/00000.npz" \
    --out logs/diagnostics/pfpp_t2a_${TS}
$G/.venv/bin/python scripts/pfpp_layout_probes.py \
    --pfpp-dir "$INF/t2_dewear/0" \
    --pc-npz   "$PF/data/pc_data/juglet_dewear/val/00000.npz" \
    --mesh-hdf5 $G/input/juglet_deploy_dewear.hdf5 --mesh-key artifact/Juglet-000 \
    --out "logs/diagnostics/pfpp_t2a_${TS}" --tag t2_dewear --n-null 0

cp $PF/logs/t2b_latent_probe/latent_probe.json logs/diagnostics/pfpp_t2a_${TS}/ 2>/dev/null || true
echo "=== DONE: $G/logs/diagnostics/pfpp_t2a_${TS} ==="
