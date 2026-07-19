#!/bin/bash
#SBATCH --job-name=pf_render_juglet_deploy
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/gpfs/projects/punim2657/Puzzlefusion/logs/render_juglet_deploy_%j.out
#SBATCH --error=/data/gpfs/projects/punim2657/Puzzlefusion/logs/render_juglet_deploy_%j.err

set -euo pipefail

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate blender-render

cd /data/gpfs/projects/punim2657/Puzzlefusion
export PYTHONPATH=$(pwd)

# .npz mesh_file_path uses artifact/Juglet-000 — point symlink at anchor-centered OBJs.
mkdir -p /data/gpfs/projects/punim2657/Dataset/artifact
ln -sfn ../Juglet_anchor_centered /data/gpfs/projects/punim2657/Dataset/artifact/Juglet-000

INFER_DIR="/data/gpfs/projects/punim2657/Puzzlefusion/output/denoiser/everyday_epoch2000_bs64/inference/juglet_deploy"
echo "Waiting for inference dir: ${INFER_DIR}"
for i in $(seq 1 120); do
  if [ -d "${INFER_DIR}" ]; then
    break
  fi
  sleep 5
done
if [ ! -d "${INFER_DIR}" ]; then
  echo "ERROR: inference dir not found after waiting: ${INFER_DIR}" >&2
  exit 1
fi

echo "=== PF++ Blender render: juglet_deploy ==="
echo "Date: $(date)"
echo "Inference: output/denoiser/everyday_epoch2000_bs64/inference/juglet_deploy/"
echo "Meshes:    /data/gpfs/projects/punim2657/Dataset/artifact/Juglet-000/ -> anchor-centered"
echo ""

rm -rf renderer/juglet_deploy_output

python -m renderer.render_results \
    experiment_name=everyday_epoch2000_bs64 \
    inference_dir=juglet_deploy \
    renderer.mesh_path=/data/gpfs/projects/punim2657/Dataset \
    renderer.num_samples=-1 \
    renderer.category=all \
    renderer.min_parts=2 \
    renderer.max_parts=20 \
    renderer.auto_frame=true \
    renderer.blender.use_GPU=false \
    renderer.blender.numSamples=64 \
    renderer.blender.imgRes_x=1024 \
    renderer.blender.imgRes_y=1024 \
    renderer.output_path=renderer/juglet_deploy_output

echo ""
echo "=== DONE ==="
echo "Output: renderer/juglet_deploy_output/0/"
