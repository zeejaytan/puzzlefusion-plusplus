# Progress log – 2026-03-23 – puzzlefusion

## Entry 1 – 18:22
- timestamp: 2026-03-23 18:22
- branch/commit: main @ 7eedadb

- what changed:
  - Fixed `renderer/render_results.py`: removed hardcoded `BlenderToolBox_render/` prefix so output writes directly to the configured path
  - Updated `renderer/render_test.slurm`: reduced to 512×512 / 32 samples / 2-hour limit; switched output_path to relative `renderer/test_output`
  - Resubmitted render job (22928800); completed successfully — 2 samples (BeerBottle fractured_20, fractured_37) rendered with `gt.png`, frames, `video.mp4`
  - Moved all render outputs into `renderer/test_output/{9,27,83}/`; deleted `BlenderToolBox_render/` directory
  - Analysed Breaking Bad dataset for pottery-sherd-like fragments: `Teacup` (avg ratio 2.7) and `Bowl` (avg ratio 1.9) identified as best categories; `Vase` (ratio ~1.1) worst
  - Created `data/pc_data/everyday/val_pottery/` — 573 symlinks (500 bowl + 73 teacup) filtered from val set by `category` field
  - Wrote `scripts/inference_pottery.slurm` and submitted SLURM job **23075382** (gpu-a100, 4h, `inference_dir=pottery_run`)

- what's next:
  - none

- blockers:
  - none
