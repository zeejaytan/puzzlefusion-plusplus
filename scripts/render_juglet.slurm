#!/bin/bash
#SBATCH --job-name=pf_render_juglet
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/gpfs/projects/punim2657/PFpp/logs/render_juglet_%j.out

set -euo pipefail

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate blender-render

cd /data/gpfs/projects/punim2657/Puzzlefusion
export PYTHONPATH=$(pwd)

# Logs and renders go to the working area, never the checkout: the run directory when
# launched through new_run.sh, otherwise one directory per job under $WORK/out/.
WORK=/data/gpfs/projects/punim2657/PFpp
OUT=${RUN_DIR:+$RUN_DIR/out}
OUT=${OUT:-$WORK/out/render_juglet_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
mkdir -p "$OUT"

# Renderer expects mesh_path + mesh_file_path.txt (artifact/Juglet-000/*.obj)
mkdir -p /data/gpfs/projects/punim2657/Dataset/artifact
ln -sfn ../Juglet /data/gpfs/projects/punim2657/Dataset/artifact/Juglet-000

echo "=== PF++ Blender render: juglet_archaeological ==="
echo "Date: $(date)"
echo "Inference: output/denoiser/everyday_epoch2000_bs64/inference/juglet_archaeological/"
echo "Meshes:    /data/gpfs/projects/punim2657/Dataset/artifact/Juglet-000/"
echo ""

python -m renderer.render_results \
    experiment_name=everyday_epoch2000_bs64 \
    inference_dir=juglet_archaeological \
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
    renderer.output_path="$OUT/juglet_output"

echo ""
echo "=== DONE ==="
echo "Output: $OUT/juglet_output/0/"
echo "  gt.png, video.mp4, imgs/*.png"
echo "Date: $(date)"
