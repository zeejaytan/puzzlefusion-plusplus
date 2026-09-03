# What we are trying to establish with PuzzleFusion++

**PF++ is a comparison partner, not a project we are developing.** It is here because it
did something the others did not: it produced a coherent Juglet shape. Two questions
follow from that, and this folder is deliberately short — a thin project honestly gets two
questions, not a forced seven.

This folder is **state, not a log**. Session notes live in [`../.progress/`](../.progress/).

Prefix **`P`**, permanent. Numbers are never reused. **P3 is next.**

| # | Question | Status | Blocked by |
|---|---|---|---|
| [P1](P1-why-does-it-form-a-shape.md) | Why does PF++ form a coherent Juglet shape when GARF and TORA do not? | **largely answered 2026-07-20** | none |
| [P2](P2-which-configuration-is-the-fair-baseline.md) | Which configuration is the fair baseline — denoiser-only or oracle? | open | none |

## What is established

| Claim | Weight | Source |
|---|---|---|
| PF++ produced a **plausible** Juglet assembly where GARF and TORA did not. | 1 object, visual | `../GARF/docs/notes/JUGLET_DEPLOY_INFERENCE_ANALYSIS.md` |
| That gap is **largely explained by inference and visualisation pipelines**, not by PF++ alone understanding the scan better. | 1 object | ibid. |
| The mechanism is the decision matrix's middle row: **the prior composes, the tokens gate plausibility**, with wear-invariance confirmed at both output and feature level. | T0–T4, 1 object, 2026-07-20 | `../GARF/docs/notes/PFPP_JUGLET_SUCCESS_FINDINGS.md` |
| The "semi-good" deploy result was **denoiser-only**, checkpoint `everyday_epoch2000_bs64`. | procedural | ibid. |

## The caveat that governs everything above

All of it rests on **one object**, the nine-piece Juglet, which has **no correct
reassembly to score against**. A striking result from a single object with no answer key is
a lead, not a conclusion. Nothing here should be written up as a general property of PF++.

## Related

`U7` (what is actually being compared) in [`../../intent/`](../../intent/);
`G1` in `../../GARF/intent/`.
