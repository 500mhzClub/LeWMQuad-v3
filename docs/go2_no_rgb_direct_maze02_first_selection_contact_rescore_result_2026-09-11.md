# Direct-model first decision: contact-horizon scoring diagnosis

The fixed first 800 observations of the running sixth comparison case show
312 selected translational arcs and maximum observed displacement of 2.0415 m
from the start, with no recorded arrival. Forward was phase-allowed in 795
observations but never selected. These are saved visual observations and
requested commands; the episode's raw physics, contact and sensor audit is
still pending.

The saved prefix was read twice identically. Its record is
`docs/go2_no_rgb_direct_maze02_live_800_observation_prefix_2026-09-11.json`,
SHA-256 `670a0c8b45dfb254df5f4d940aa6df10d6506e5fb7c10fd6a672457c2a9a253d`.
Its canonical original 800-row identity is
`ef3740c2539fd8284eaa88f68bdf9f58935adbdd28fea770ee136f352a3e792a`.

Applying the already prepared ordinary commitment-contact scorer to the saved
first selection, frame 3, changes left arc `[0.16, 0, 0.45]` to forward
`[0.2, 0, 0]`. The scorer reconstructs and authenticates the entire original
selection before changing its contact-cost horizon from 800 ms to 100 ms.
The pose/progress horizon is already 100 ms. All forecasts, residual receipts,
phase restrictions, surface checks and 800 ms nominal-path vetoes remain
identical; the original selection is unmodified.

| Candidate | Original utility (m) | 100 ms contact utility (m) |
| --- | ---: | ---: |
| Forward | 0.003047 | 0.020816 |
| Left arc | 0.009489 | 0.015914 |
| Right arc | 0.003151 | 0.015284 |

The result record is
`docs/go2_no_rgb_direct_maze02_first_selection_contact_rescore_2026-09-11.json`,
SHA-256 `4f600438712076e9d745c88e53279859feba3738b4b7715d40fe1dabb5c8dcec`.
Execution session 22601 exited 0. All 1,988 source bindings and both input
record identities were checked before and after the calculation.

This is a post-hoc scoring calculation on one saved decision. It ran no model
inference, collected no new observations and executed no command. It makes no
claim about later counterfactual motion or navigation success. The contact
scores are uncalibrated. The existing prospective contact-scoring pilot uses
the full supervised model; its case and queue position remain unchanged.
The finding supports testing the existing contact-horizon hypothesis across
model heads if prospective navigation results justify that follow-up.
