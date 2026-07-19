#!/bin/bash
#SBATCH --job-name=blender_pottery
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data/gpfs/projects/punim2657/Puzzlefusion/logs/render_pottery_%j.out
#SBATCH --error=/data/gpfs/projects/punim2657/Puzzlefusion/logs/render_pottery_%j.err

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate blender-render

cd /data/gpfs/projects/punim2657/Puzzlefusion
export PYTHONPATH=$(pwd)

python -m renderer.render_results \
    experiment_name=everyday_epoch2000_bs64 \
    inference_dir=pottery_render_selection \
    renderer.num_samples=-1 \
    renderer.blender.use_GPU=false \
    renderer.blender.numSamples=64 \
    renderer.blender.imgRes_x=1024 \
    renderer.blender.imgRes_y=1024 \
    renderer.output_path=renderer/pottery_output
