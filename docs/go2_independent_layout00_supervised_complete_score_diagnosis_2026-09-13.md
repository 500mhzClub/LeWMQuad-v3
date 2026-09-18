# Supervised layout 0: complete score-induced stall diagnosis

The saved supervised collection contains 8,014 decisions: 8,000 navigation
selections and 14 observations without a selection. A single pass over the
complete closed decision stream reconstructed every navigation score and
selected maximum. This is recorded-decision analysis; full raw sensor and
controller replay audit was still running when it completed.

All six candidate actions passed the recorded phase, sampled-surface and
nominal-path feasibility filters on every one of the 8,000 navigation choices.
The selected actions were 7,813 holds (97.6625%) and 187 left turns (2.3375%).
There were zero nonzero translation requests. The hold preference persisted
throughout the run, with 975–977 holds in each complete thousand-frame block.
There were no empty feasible banks or unexplained overrides of the score
maximum. The 14 observations without selection remain explicitly counted.

On all 7,813 hold decisions, forward had a positive progress-score advantage
over hold, but its additional contact penalty was larger. Across these choices:

| Forward compared with hold | Mean score contribution |
| --- | ---: |
| Additional progress/alignment score | +0.014298 m |
| Additional contact penalty | −0.053399 m |
| Resulting utility disadvantage | −0.039101 m |

These are utility terms expressed in metres, not measured displacement or
calibrated collision probabilities. The score combines 100 ms progress with
an 800 ms contact term. Geometry vetoes did not eliminate forward motion in
this episode; the recorded ranking selected hold despite its feasibility.
The earlier native trace readout independently found no arrivals, no crossings,
no recorded contacts and less than 1 cm displacement from the start.

This strengthens the diagnosis behind the already queued commitment-contact
JEPA/supervised pair. That pair was fixed before this complete-stream analysis
and is unchanged. Its native results must determine whether matching the
contact horizon to the executed interval improves movement and mission
success. A feasible candidate is not a physical safety certificate, and this
analysis cannot establish the outcome of an unexecuted action.

The diagnostic completed in 88.790 seconds, session 55335, exit code 0.
No model inference, controller replay, alternative rescoring or native execution
was performed. Source:
`scripts/summarize_supervised_layout00_action_scores_development.py`.
The complete counters, exclusions, per-thousand-frame action counts, score
statistics and input identities are in
`go2_independent_layout00_supervised_complete_score_diagnosis_2026-09-13.json`.
The collection result SHA-256 remains
`138d4c5578f122f73662d2fac8bc4960677149792de727f68ceeb724173a6ec5`;
the decoded complete decision stream SHA-256 is
`4ac3c0277d202d063655346f27b27589759a51e61b50a71f690ac01224f0f77c`.
