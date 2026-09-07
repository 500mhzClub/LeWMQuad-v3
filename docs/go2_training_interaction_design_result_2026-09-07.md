# Training interaction diagnostic: diverse graphs, nearly repeated local outcomes

## Result and consequence

The complete six-layout **training** population has no strict reversal of any
action pair's contact ordering within a history/support condition. All 120
six-action groups share zero observed contact for actions 2–5. Contact vectors
are identical across the six layouts for each context/history/support cell.
This confirms the contact-ranking shortcut on training data, not just the
previously exposed development evaluation.

The next learning collection needs scene-dependent interactions and useful
progress. More optimizer updates or more topologically distinct layouts with
the same local excitation would not by themselves address the demonstrated
coverage weakness. This is a design diagnosis, not proof that no fitted model
could improve motion prediction on the current data.

The immediate execution experiment remains the already frozen independent
tracking challenge from the [result and next-steps plan](go2_independent_pulse_parallel_result_and_next_steps_2026-09-07.md).
This diagnostic adds no new prerequisite, native probe, training run or reason
to wait for a positive JEPA result. Its storage gate still requires about
4 GiB more space; deletion approval remains unanswered.

## Actual data and arithmetic

Read only the terminal-bound metadata products for training batches l00–l05:
720 departures, 120 matched six-action groups and one fixed two-second horizon.
Reconstructed the existing exact-prefix/contact coverage and required equality
with the original completed study report. Selection/development target products,
RGB, depth images, native traces and checkpoints were not loaded by this diagnostic.
The inventory contains development role identities, but no new development labels
were consumed here. Previous development results were already exposed, so this
is a post-hoc training diagnostic, not an untouched prospective test.

| Observed contact vector, actions 0–5 | Training groups |
| --- | ---: |
| 0, 0, 0, 0, 0, 0 | 102 |
| 0, 1, 0, 0, 0, 0 | 6 |
| 1, 1, 0, 0, 0, 0 | 12 |

The six one-positive groups are near-wall/quiet/nominal. The twelve
two-positive groups are near-wall/recent-forward under the two support
conditions. Each pattern repeats across all six layouts. For each of the
four history/support strata, all 15 action pairs were examined: none has
one action strictly better in one group and strictly worse in another.
Ties do not count as reversals. The 60 pair/stratum checks are descriptive,
not independent statistical trials.

The command vocabulary explains why the common zero-contact set is insufficient:

| Actions | Requested velocity (forward m/s, lateral m/s, yaw rad/s) | Pulse durations |
| --- | --- | --- |
| 0, 1 | (0.2, 0, 0) | 0.2 / 0.5 s |
| 2, 3 | (0, 0, +0.45) | 0.2 / 0.5 s |
| 4, 5 | (0, 0, −0.45) | 0.2 / 0.5 s |

Each pulse is followed by braking. At the common two-second horizon, the
collision-free observed forward displacement ranges were 9.26–37.88 mm for
action 0 and 30.51–107.90 mm for action 1. These exclude 12 and 18 censored
motion outcomes respectively. All 120 motion targets were observed for each
turn action. Across those turns, maximum absolute measured yaw was approximately
0.249 rad (14.3 degrees), not a maze-scale 90/180-degree turn. Turning can produce
incidental translation, so zero commanded forward velocity is not zero actual
displacement. No goal-directed utility was defined or evaluated here.

For each action/context/history/support cell, calculate the RMS XY deviation
of its available motion labels about that cell's cross-layout mean. Of 120
planned cells, 115 have all six observed labels and five have none because
all six outcomes are terminal-censored. The missing cells remain explicit,
not silently treated as zero error. Among the 115 observed cells:

- Mean cell RMS: **0.261 mm**.
- Median cell RMS: **0.067 mm**.
- Maximum cell RMS: **1.255 mm**.

These are within-training label dispersions, **not held-out prediction errors**,
nor isolated causal effects of geometry. Conditioning uses construction context,
history and support labels; those labels are not deployment sensor inputs.
Sensor histories across different scenes have not been shown identical. The
comparison nevertheless reveals that this experiment's topological diversity
produces little additional variation in these short, observed local endpoints
once those conditions are fixed.

## Implementation and verification

- [Pure diagnostic](../lewm/pulse_training_interaction_diagnostic_development.py):
  requires the complete supplied training-group roster, rejects nontraining
  roles and malformed labels, distinguishes observed-subset from full-population
  claims, handles ties and keeps missing-prefix/censored groups in denominators.
  Even a positive reversal result leaves RGB contribution and navigation false.
- [Read-only adapter](../scripts/read_go2_independent_pulse_training_interaction_v1.py):
  verifies the exact completed study terminal, launch/dataset/coverage bindings,
  inventory and each training audit/product identity before parsing, and verifies
  the consumed bindings again afterward. It reconstructs target clocks/masks
  and existing coverage using the original dataset/evaluation implementations.
  It does not independently rerun raw auditing or reauthenticate all study
  tensors; that broader completed-study check was performed previously.
- [Synthetic tests](../lewm/tests/test_pulse_training_interaction_diagnostic_development.py):
  **15 passed in 0.10 s**, including constant-turn shortcuts, genuine reversals,
  support separation, ties, all-contact/all-missing data, missing planned groups,
  invalid/nontraining rows and 20 random populations checked against a separate
  brute-force group-pair definition. These tests are arithmetic checks, not new
  scientific evidence.

The first diagnostic execution completed normally. Inspection found that its
motion summary omitted wholly censored cells; the reporting loop was corrected
to enumerate all 120 planned cells, then the same read-only diagnostic completed
again. No data, model, thresholds or experiment were changed. Both invocations
were metadata diagnostics, not repeated physical attempts.

Full result: [training diagnostic JSON](go2_independent_pulse_training_interaction_diagnostic_2026-09-07.json),
SHA-256 `0b3011f0fa25977b8290a6a91ad59ee85d21df32462fe20a36958eee3f2af056`.
It retains all group labels, all pair witnesses, all 120 dispersion cells,
action-level target denominators, consumed metadata bindings and diagnostic
source identities. The trained-study terminal remains
`588f24def6ec8810ae5a3411277576b0d965c77bf6ffdb8e18cfd80dce7b8122`.

## Implications for the successor, without claiming success

Use balanced visible obstacle/branch configurations in which different
progressing actions are appropriate. Include executed motion far enough to
interact with the relevant geometry and separately test successful braking and
maze-scale turning. Preserve stop behavior and physical supervision; do not
extend an unsafe terminated tape to obtain a convenient latent target.

Define goal progress before collecting/scoring. An absence of strict contact
reversals does not prove that visual sensing is unnecessary for all tasks;
conversely, adding risk reversals alone can still leave a third constant action
that solves every case. Audit both common-action shortcuts and progress, then
test sensor-conditioned selection in actual closed-loop missions. Use complete
training-only coverage to diagnose the prospective collection, not select a
favorable evaluation subset. Keep explicit terminal-event and future-image
support as specified in the next-steps plan.

No training, tracking integration, mission completion, online memory benefit,
real-time or hardware result has been added by this diagnostic. Those remain
the required downstream work, not optional replacements for prediction analysis.
