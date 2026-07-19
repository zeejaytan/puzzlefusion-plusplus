#!/bin/bash
#SBATCH --job-name=pf_render_test
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=render_test_%j.log

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate blender-render

cd /data/gpfs/projects/punim2657/Puzzlefusion
export PYTHONPATH=$(pwd)

echo "Starting renderer test — $(date)"
echo "Sample: data_id 97 (Bottle, 13 frags, 84.6% acc)"

python renderer/render_results.py \
    experiment_name=everyday_epoch2000_bs64 \
    inference_dir=test_run \
    renderer.mesh_path=/data/gpfs/projects/punim2657/Breaking-Bad-Dataset.github.io/data/breaking_bad/ \
    renderer.output_path=render_test \
    renderer.num_samples=1 \
    renderer.category=Bottle \
    renderer.min_parts=13 \
    renderer.max_parts=13 \
    renderer.blender.use_GPU=False \
    renderer.blender.numSamples=16 \
    renderer.blender.imgRes_x=512 \
    renderer.blender.imgRes_y=512

echo "Done — $(date)"
echo "Output: BlenderToolBox_render/render_test/"
ls BlenderToolBox_render/render_test/ 2>/dev/null || echo "Output dir not found"
