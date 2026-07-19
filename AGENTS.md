# AGENTS.md — PuzzleFusion++ (project)

Follow the workspace root **`../AGENTS.md`** (laptop ↔ GitHub ↔ Spartan) for all shared rules. This file only adds PuzzleFusion++-specific paths and domain notes.

## PuzzleFusion++ paths

| Role | Value |
|------|--------|
| GitHub fork (`origin`) | `zeejaytan/puzzlefusion-plusplus` |
| Upstream | `eric-zqwang/puzzlefusion-plusplus` |
| Spartan checkout (`REMOTE_ROOT`) | `/data/gpfs/projects/punim2657/Puzzlefusion` |
| SSH | `Host spartan`, user `zhuojiat` |
| Remote helpers | `scripts/remote/pull_and_sbatch.sh`, `job_status.sh`, `fetch_artifacts.sh` |
| Breaking Bad meshes (HPC only) | `/data/gpfs/projects/punim2657/Breaking-Bad-Dataset.github.io/data/breaking_bad/` |

Heavy data on Spartan only (gitignored): `data/`, `output/`, `logs`, `checkpoints.zip`, `*.hdf5` / `*.h5`, `pytorch3d/`, `chamferdist/`, render outputs. Local rsync landing zone: `artifacts/`.

Typical loop:

```bash
git push origin HEAD
./scripts/remote/pull_and_sbatch.sh scripts/eval_juglet.slurm
./scripts/remote/job_status.sh
./scripts/remote/fetch_artifacts.sh logs/some_run ./artifacts/
```

## Domain / debugging

See `CLAUDE.md` for the full codebase reference (VQVAE / SE3 denoiser / verifier modules, configs, data layout). Prefer probes over guesses; do not invent config keys. Session notes live in `.progress/`.
