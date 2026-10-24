# 01: Build pytorch3d and chamferdist outside the repo; logs and renders out of it

**What to build:** The compiled dependencies live under the working area's `envs/` or `build/`, renders and logs go to run directories, and the repo checkout holds source only.

**Answers:** none (workbench routine).

**Blocked by:** None (can start immediately).

**Status:** resolved — both trees in PFpp/build, env proved on a GPU node (job 33745440)

Amendments 2.6 and 2.3. Rebuilding `pytorch3d` is slow: move it (and point the environment at the new path) rather than rebuild, and prove the environment still imports it on a compute node.

- [x] `pytorch3d/` and `chamferdist/` no longer inside the checkout; imports still work
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
- Move (conservator's OK on the decision card, 2026-10-24 08:50Z): `move_builds_out.sh` dry run
  clean, then applied at 90553f1; both trees now in `PFpp/build/`, `pytorch3d.egg-link` and
  `easy-install.pth` repointed (originals in `PFpp/attic/env-backup-*`, record in
  `PFpp/attic/move-builds-*.txt`); imports from the new place on the login node.
- Compute-node proof: job 33745440, COMPLETED 0:0, 1 min 19 s on an A100 (gpgpu127). torch
  2.0.1+cu118 with CUDA; pytorch3d 0.7.9 loaded from `PFpp/build/pytorch3d`, chamferdist from
  the env egg; `knn_points` and Chamfer both ran on the GPU; log ends "ALL IMPORTS OK"
  (`PFpp/logs/env_imports_33745440.out`).

**Not in this ticket, still inside the checkout:** `output/` (trained models and inference
results, written by Hydra relative to the working directory), `data/` (the npz builders write
there), the old `logs/` (6 MB) and render folders under `renderer/` (608 MB) already on disk,
which notes may cite, and `generate_viz.py`'s default `assembly_viz.html`. Moving those needs
`project_root_path` split from the code path; a separate ticket.
