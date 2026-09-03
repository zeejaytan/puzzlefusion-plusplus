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

## Agent skills

Configured here so this repo works when opened on its own, not only from the `C:\PR`
umbrella. The full text of each convention lives at the workspace root; these are the
parts an agent needs before it can act.

- **Issue tracker — local markdown.** One feature per directory: the spec at
  `.scratch/<feature>/spec.md`, tickets one per file at
  `.scratch/<feature>/issues/<NN>-<slug>.md`, numbered from `01` in dependency order.
  Every ticket carries an **`Answers:`** line naming the question in `intent/` it exists
  to settle -- `P1` for this project, `U6` for the workspace, or `none` for routine
  work. Conventions and the ticket template: `../docs/agents/issue-tracker.md`.
- **Triage labels.** `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`,
  `wontfix`, recorded as a `Status:` line near the top of the ticket. Details:
  `../docs/agents/triage-labels.md`.
- **Domain docs — single-context.** This file is the domain reference for this project;
  ADRs, if any, go under `docs/adr/`. Details: `../docs/agents/domain.md`.
- **Intent.** [`intent/`](intent/) holds what we are trying to establish and what would
  settle it -- prefix **`P`**, permanent, numbers never reused. `/to-intent` opens a
  question or writes a finished ticket's result back into one. Check the loop is wired
  with `python ../scripts/check_intent_links.py`.

**Do not run `/setup-matt-pocock-skills` in this repo.** It would replace the above with
its own defaults, and its ticket template has no `Answers:` line -- tickets would stop
being connected to the question they exist to answer, silently.
