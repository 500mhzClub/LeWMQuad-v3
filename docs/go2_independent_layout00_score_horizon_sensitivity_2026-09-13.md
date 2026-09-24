# Contact-horizon sensitivity across an early independent-maze prefix

The diagnostic covers all 701 completed recorded decisions at frames 200–900
of the live layout-0 full-RGB JEPA case. All 701 used execution-time intermediate
waypoint scoring and reported an observed-floor route to a frontier.

The original score and selected action reconstruct in 698 decisions from the
recorded candidate terms and feasible bank. Three decisions had no candidate in
that bank and were excluded from the ranking comparison. There were no score
algebra mismatches or unexplained final selections in the nonempty banks.

The diagnostic keeps the recorded forecasts, residual-corrected distance and
alignment progress, contact penalty, action order and feasibility filters. It
changes only the contact term from the recorded 0.8-second cumulative score to
the recorded 0.1-second commitment score, then computes the highest-ranked
action on that same bank.

- Rankings change in 504 of 698 reconstructed decisions (72.2%).
- Of 420 recorded turns in this population, 270 (64.3%) would instead rank a
  translating action first.
- In the full 701-decision window, the actual commands comprised 423 turns,
  262 translating actions and 16 holds.
- The recorded frontier target changed 37 times.
- Observed goal distance increased from 2.240753 m at frame 200 to 2.895360 m
  at frame 900. Both endpoints were outbound with no terminal state. Distance
  to the final goal alone does not measure successful exploration.

This supports investigating the interaction between short-horizon progress
reward and longer-horizon contact penalty. It does not establish that the
shorter contact horizon is correct, safe or better for navigation. These are
uncalibrated model scores; long-horizon risk may be useful. Frontier selection,
physical dynamics and subsequent observations could change the result of a
prospective intervention.

No model inference, controller replay, native simulation or command execution
was performed by the diagnostic. No alternative trajectory was inferred, and
the live JEPA run and queued controls are unchanged. The next causal evidence
comes from completed prospective comparisons; these rerankings are not counted
as navigation successes or independent maze replicates.

The diagnostic completed in session 44179. Source:
`scripts/diagnose_independent_layout00_score_horizon_development.py`.
Exact counts, transitions, examples, scope flags and the decoded recorded-prefix
hash are in `go2_independent_layout00_score_horizon_sensitivity_2026-09-13.json`.

Subsequent physical outcome: the unchanged JEPA controller completed collection
with outbound arrival at 2061 and return arrival at 3439. The existing native
evaluator passes both dwell windows, physical backtracking and terminal quiet,
with zero recorded contact flags. The full audit remains pending; see
`go2_independent_layout00_jepa_physical_readout_2026-09-13.md`. Thus the early
turning and ranking sensitivity did not prevent this physical candidate pass.
They are not grounds for labeling this run a navigation failure or changing
the queued comparison policies. Comparative efficiency and any separate
prospective scoring intervention remain distinct questions.

The subsequent supervised-rollout case provides a separate development
observation. At its completed decision 200, it selected hold while a waypoint
was available and the mission did not require stopping. The first 201 records
contain three initial no-action decisions, three left turns and 195 holds.
All six phase candidates have no sampled surface conflict and all eight
predicted segments are nominally clear for every candidate.

| Candidate at supervised decision 200 | Executed distance + alignment progress (m) | Full-plan contact score | Final utility (m) |
| --- | ---: | ---: | ---: |
| Hold | 0.00018205 | 0.00340829 | -0.00390790 |
| Forward | 0.01444957 | 0.04879320 | -0.04410227 |
| Left arc | 0.01039418 | 0.02373762 | -0.01809097 |

The 1.2 m contact coefficient makes the forward contact term approximately
0.05855 m. This is the same 100 ms progress / 800 ms contact scoring contract
as the JEPA case; the contact scores are uncalibrated. The observation points
to model–planner scoring interaction rather than a geometric rejection at this
frame. It does not prove that the complete supervised run will remain still,
or that changing the score would yield successful navigation. The live case
and queued controls retain their original settings and 8,000-decision budget.

Later completed supervised snapshots remain near the start: decision 1500
reports position (0.000750, 0.007916) m and goal distance 2.592084 m;
decision 2000 reports position (0.001181, 0.007781) m and goal distance
2.592219 m. Both request zero motion, remain outbound, and record no arrival
or terminal state. These are sampled observed poses, not a claim that every
intervening command was zero or a final native-verified failure. The original
process remained live when decision 2000 was inspected. Separate prospective
commitment-contact JEPA and supervised cases are queued after the unchanged
baseline comparisons; see `go2_commitment_contact_scoring_experiment_2026-09-13.md`.

The completed decision-3000 snapshot likewise reports a zero request, outbound
phase, no arrivals and no terminal state. Its observed position is
(0.000717, 0.008103) m and goal distance is 2.591897 m. This extends the sampled
near-start behavior to 300 seconds of simulated control time, while the original
process continues its fixed budget. Final physical and replay results remain
pending.

A read-only rescore of that decision-3000 selection confirms that the same
scoring interaction persists at this later observation. All six original
candidates remain feasible. Original hold and forward utilities are
-0.003846695 m and -0.043004553 m, respectively. Applying the already prepared
100 ms contact term gives -0.000009216 m and +0.013911560 m and ranks forward
first. Raw forecasts, geometry filters and causal residual evidence are
unchanged. No checkpoint was loaded and no alternative controller trajectory
was executed or inferred; this supports the existing prospective experiment
without establishing a navigation benefit or changing its settings.

At supervised decision 3500, the observed position is (0.000038, 0.008365) m,
goal distance is 2.591635 m, and the command remains zero with no arrival or
terminal state. For context, the unchanged JEPA case on the same independent
layout had a verified return arrival at decision 3439 and completed its terminal
quiet interval by decision 3449; see
`go2_independent_layout00_jepa_verified_round_trip_2026-09-13.json`.
This is an interim within-layout behavior contrast, not a final supervised
failure, a verified supervised physical result, or cross-layout evidence of
JEPA superiority. The supervised case retains the full 8,000-decision budget.
