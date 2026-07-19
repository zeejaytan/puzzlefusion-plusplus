# Progress log – 2026-03-27 – puzzlefusion

## Entry 1 – 12:59
- timestamp: 2026-03-27 12:59
- branch/commit: main @ 7eedadb

- what changed:
  - Fixed `renderer/render_results.py`: removed hardcoded `BlenderToolBox_render/` prefix so output writes to the path given directly
  - Fixed `renderer/myrenderer.py`: moved `import bpy` to top so `mathutils` submodule is available at import time
  - Updated `config/auto_aggl.yaml`: set `mesh_path` to absolute Breaking Bad dataset path
  - Analysed Breaking Bad dataset categories for sherd-like fragment geometry; identified `bowl` and `teacup` as best pottery analogues
  - Created `data/pc_data/everyday/val_pottery/` — 573 symlinks filtered to bowl+teacup samples only
  - Created `scripts/inference_pottery.slurm` and ran inference (job 23075382); completed in 16:17, all 573 samples
  - Pottery inference results: part_acc=0.669, rmse_r=32.1°, rmse_t=0.103, shape_cd=0.00424
  - Selected 9 representative samples (diverse category/parts/accuracy) into `inference/pottery_render_selection/`
  - Created `scripts/render_pottery.slurm`; submitted render job 23150083 (running, 1024×1024, 64 samples/frame)

- what's next:
  - none

- blockers:
  - none
