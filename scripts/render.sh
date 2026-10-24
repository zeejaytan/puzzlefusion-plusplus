#!/usr/bin/env bash
# Renders go to the working area, never the checkout.
OUT=${OUT:-/data/gpfs/projects/punim2657/PFpp/out/render_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"

python renderer/render_results.py \
    experiment_name=everyday_epoch2000_gpu4_bs64 \
    inference_dir=denoiser_only \
    renderer.num_samples=20 \
    renderer.output_path="$OUT" \
    renderer.blender.imgRes_x=512 \
    renderer.blender.imgRes_y=512 \
    renderer.blender.numSamples=50
