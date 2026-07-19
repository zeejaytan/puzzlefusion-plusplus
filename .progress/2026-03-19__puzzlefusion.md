# Progress log – 2026-03-19 – puzzlefusion

## Entry 1 – 14:35
- timestamp: 2026-03-19 14:35
- branch/commit: main @ 7eedadb

- what changed:
  - Investigated `renderer/` folder: uses `bpy`, `blendertoolbox`, `mathutils` to animate and render mesh fragments via Blender, outputs PNG frames + `video.mp4` via ffmpeg
  - Confirmed no existing Blender module on Spartan; all pip `bpy` versions require Python 3.11 (existing envs are 3.8/3.9)
  - Created new conda env `blender-render` (Python 3.11)
  - Installed `bpy==4.2.0`, `blendertoolbox==0.0.9`, `scipy`, `hydra-core==1.3.2` via pip; `ffmpeg` via conda-forge
  - Verified `bpy` and `mathutils` imports work; confirmed `renderer/myrenderer.py` import order was already correct
  - Created `renderer/render_test.slurm` — test job (2 samples, CPU, `sapphire` partition) using `blender-render` env
  - Submitted SLURM job 22925788; results will appear at `renderer/test_output/`

- what's next:
  - none

- blockers:
  - none
