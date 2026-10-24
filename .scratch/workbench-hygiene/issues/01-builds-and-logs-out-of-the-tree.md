# 01: Build pytorch3d and chamferdist outside the repo; logs and renders out of it

**What to build:** The compiled dependencies live under the working area's `envs/` or `build/`, renders and logs go to run directories, and the repo checkout holds source only.

**Answers:** none (workbench routine).

**Blocked by:** None (can start immediately).

**Status:** in-progress — scripts changed; move on Spartan next
**Working in:** project thread "Puzzlefusion builds out of tree" (2026-10-24)

Amendments 2.6 and 2.3. Rebuilding `pytorch3d` is slow: move it (and point the environment at the new path) rather than rebuild, and prove the environment still imports it on a compute node.

- [ ] `pytorch3d/` and `chamferdist/` no longer inside the checkout; imports still work
- [x] No batch script writes logs or renders into the checkout

## Progress

**2026-10-24.** Working area chosen: `/data/gpfs/projects/punim2657/PFpp/` (a sibling of the
checkout, as `SfSpp/` is for SfS++), holding `build/`, `logs/`, `out/`, `runs/`, `attic/`.

- Spartan survey (read-only): `pytorch3d/` 813 MB is an editable install (`pytorch3d.egg-link`
  plus a line in `easy-install.pth` in `~/.conda/envs/puzzlefusionpp`, Python 3.8), so moving it
  needs those two files repointed. `chamferdist` 1.0.3 is a normal egg in site-packages: the
  70 MB `chamferdist/` source tree is not imported at all. No other environment points at either.
- All 19 `.slurm` files: Slurm output to `PFpp/logs/<name>_%j.out`, stderr merged in (when
  launched through `new_run.sh` its `--output` overrides this with `<run>/logs/`). Renders and
  probe outputs go to `$RUN_DIR/out` or else `PFpp/out/<name>_<jobid>/`; the T2/T3 jobs also
  stopped writing into GARF's checkout. `make_mesh_gif.sh`, `render_sample.sh` and
  `scripts/render.sh` likewise. `install_setup.slurm` and `fix_torch.slurm` build in `PFpp/build/`.
- `scripts/move_builds_out.sh` (dry run unless `--apply`) moves both trees and repoints the
  environment, backing up each edited file to `PFpp/attic/env-backup-<stamp>/`;
  `scripts/check_env_imports.slurm` runs a CUDA op from each on a compute node.

**Not in this ticket, still inside the checkout:** `output/` (trained models and inference
results, written by Hydra relative to the working directory), `data/` (the npz builders write
there), the old `logs/` (6 MB) and render folders under `renderer/` (608 MB) already on disk,
which notes may cite, and `generate_viz.py`'s default `assembly_viz.html`. Moving those needs
`project_root_path` split from the code path; a separate ticket.
