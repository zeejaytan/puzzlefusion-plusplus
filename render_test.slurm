#!/bin/bash
#SBATCH --job-name=pf_render_test
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/data/gpfs/projects/punim2657/PFpp/logs/render_test_%j.out

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate blender-render

cd /data/gpfs/projects/punim2657/Puzzlefusion

# Logs and renders go to the working area, never the checkout: the run directory when
# launched through new_run.sh, otherwise one directory per job under $WORK/out/.
WORK=/data/gpfs/projects/punim2657/PFpp
OUT=${RUN_DIR:+$RUN_DIR/out}
OUT=${OUT:-$WORK/out/render_test_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
mkdir -p "$OUT"
export PYTHONPATH=$(pwd)

echo "Starting renderer test — $(date)"
echo "Sample: data_id 97 (Bottle, 13 frags, 84.6% acc)"

python renderer/render_results.py \
    experiment_name=everyday_epoch2000_bs64 \
    inference_dir=test_run \
    renderer.mesh_path=/data/gpfs/projects/punim2657/Breaking-Bad-Dataset.github.io/data/breaking_bad/ \
    renderer.output_path="$OUT/render_test" \
    renderer.num_samples=1 \
    renderer.category=Bottle \
    renderer.min_parts=13 \
    renderer.max_parts=13 \
    renderer.blender.use_GPU=False \
    renderer.blender.numSamples=16 \
    renderer.blender.imgRes_x=512 \
    renderer.blender.imgRes_y=512

echo "Done — $(date)"
echo "Output: $OUT/render_test/"
ls "$OUT/render_test/" 2>/dev/null || echo "Output dir not found"
