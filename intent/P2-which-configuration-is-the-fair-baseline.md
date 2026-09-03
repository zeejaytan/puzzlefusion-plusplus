# P2 — Which configuration is the fair baseline?

**Status:** open · **Blocked by:** none · **Effort:** a decision, not a run
**Feeds:** [`../../intent/U7`](../../intent/U7-what-is-being-compared.md)

## Why it matters

PF++ has been evaluated in two configurations, and the comparison table carries both:
**denoiser-only** and **oracle (full pipeline)**. They are not interchangeable. An oracle
arm is given information the deployed system would not have, so a table that puts it
beside GARF and TORA is comparing three things of which one has been helped.

The Juglet result everyone cites is the **denoiser-only** one — the weaker configuration —
which is the honest side of this. But the table shows both, and a reader taking the best
column from each row would produce a comparison that no system could reproduce.

## Done when

- [ ] One configuration named as **the** PF++ baseline, with the reason written down
- [ ] Every comparison table states which configuration each column is, in the table, not
      in a footnote
- [ ] The oracle arm labelled as an **upper bound**, not as a result — it answers "how far
      could this go if the hard part were solved", which is a useful and different question

## Gate

If the two configurations differ by more than the differences between systems, then the
system comparison is not measuring what it claims to and the table should not be published
in that form.

## Source

`../GARF/docs/notes/GARF_vs_PuzzleFusion_comparison.md` (2026-04-08 denoiser-only,
2026-04-15 oracle); `../../intent/U7-what-is-being-compared.md`.
