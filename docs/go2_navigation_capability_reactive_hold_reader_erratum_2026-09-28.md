# Reactive-selector hold reader: evaluator erratum, 28 September 2026

**What happened.** Capability qualification stopped launching new missions at about 17:30 BST. The frozen V4 episode reader crashed with `KeyError('utility_m')` on C2 missions 10/0 and 11/0. Both missions recorded normally (`EPISODE_RECORDED`, terminal at 230.5 s and 406.8 s simulated). Only scoring failed.

**Cause.** The reader's hold-taxonomy classifier (`scripts/analyse_go2_stage_a_holds_readonly_development.py:classify`) assumes the prediction-based selector's record format: each candidate has a `utility_m`, plus forecast-clearance records. C2's reactive selector logs candidates as `{action, eligible, normalized_command_distance_squared}`. The C2 pilot had zero holds, so this path had never been exercised. It is a latent evaluator incompatibility, not a harness, controller or safety defect.

**Correction.** This is evaluator-only. No harness, bound or controller file changes.

- **New reader wrapper.** `scripts/read_go2_capability_v4_reactive_holds_development.py` wraps the frozen reader. It routes only rows that have no `scan_utilities` and a candidate lacking `utility_m` to a reactive classification. Those are exactly the rows that previously raised. Every other row goes to the unchanged classifier and override labels.
- **Reactive classification**, in order:
  - an explicit `heading_first_terminal` change to hold is an explicit override, `HEADING_FIRST_TERMINAL_HOLD`, labelled "planned arrival settling";
  - otherwise, no eligible movement is "no eligible movement" (the reactive log does not name the excluding rule);
  - hold nearest the desired command among eligible candidates is "movement outscored";
  - anything else is "insufficient evidence".
- **What doesn't change.** Arrival, contacts, clearance, SPL, stall rate and timing computations are unchanged.
- **Where it applies.** The wrapper is used for C2 episodes with at least one frame. All other controllers keep the frozen reader. The results are identical by construction and by the containment check.

**Containment.** The new classifier and labels reproduce the stored hold classification of every already-scored V4 episode exactly: **76 episodes and 7,092 hold rows, 0 mismatches**. None of those rows takes the reactive path.

**C2 outcome of the correction.** All 11 holds in 10/0 and 11/0 are the explicit terminal hold at the goal (observed distance about 0.020 m), labelled planned arrival settling.

**Resumption.** The validation owner resumes with `--resume`. It closes out 10/0 and 11/0 with the wrapper, then launches the unstarted assignments unchanged. No mission is rerun.
