#!/bin/bash
#SBATCH --job-name=blender_render_test
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/data/gpfs/projects/punim2657/Puzzlefusion/renderer/render_test_%j.out
#SBATCH --error=/data/gpfs/projects/punim2657/Puzzlefusion/renderer/render_test_%j.err

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate blender-render

# Run from project root so hydra resolves paths correctly
cd /data/gpfs/projects/punim2657/Puzzlefusion

python -m renderer.render_results \
    experiment_name=everyday_epoch2000_bs64 \
    inference_dir=test_run \
    renderer.blender.use_GPU=false \
    renderer.num_samples=2 \
    renderer.blender.numSamples=32 \
    renderer.blender.imgRes_x=512 \
    renderer.blender.imgRes_y=512 \
    renderer.output_path=renderer/test_output
