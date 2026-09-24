# Maze2 stops between observed clearance and forecast feasibility

The second independent-layout collection stopped because no candidate satisfied
the existing nominal constraints. Its saved full raw audit passes sensor
reconstruction, model/controller/command replay, unchanged model state and strict
physical visibility, with no hard measurement failures and no arrival window.
At this report's final inspection the worker is still checking final bindings;
its terminal and cohort progress_after_02 artifacts are not yet published.

Root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_independent_floor_transport_mazes_v1_attempt_001`.

| Artifact | SHA-256 |
| --- | --- |
| full_jepa_novel_maze_02/result.json | 50f932d68cdc011aacf36de724329dc29c3a674a032827770fe6863b6ce3855c |
| full_jepa_novel_maze_02_audit.json | 71e3aaaa27af0f95d26c3e3a7960094578ca5fdf0e7b0c0231bda1ee5c0a2798 |
| full_jepa_novel_maze_02/context_decisions.jsonl.gz | d8ad4eefbfae066634932c5f3c95ff9420ee8b3db52fe8f9502291d67814f106 |

The54,219,039-byte decision stream was rehashed before/after the final bounded
inspection4194 and remained unchanged. Collection contains514paired
observations/decisions,513completed commands,26,400physics samples and10zero
drain commands. Physical and acquisition stops are null. The terminal is
`NO_PHASE_CANDIDATE_SATISFYING_SURFACE_AND_NOMINAL_CONSTRAINTS`; no controller
failure string is recorded. This is distinct from maze1's visual-registration
failure and maze0's long return-ranking stall.

## Recorded infeasible intervals

Infeasible selections occurred at463,484–491and493–503. The controller
recovered after the first two intervals; at492it selectedright_turn
`[0,0,-.45]`. The final interval reaches11consecutive infeasible observations
at503and triggers the existing stop. The mission remains OUTBOUND, with
observed target distance3.9472608257675637m at503and no arrival.

All six candidate actions passed the sampled articulated-surface checks at
493and503, including the measured floor-contact handling. All failed their
first100ms nominal connector and therefore also failed full800ms nominal
path admission. The nearest observed occupied square was cell`[13,11]`.

| Action | First-step minimum clearance at493 (m) | At503 (m) |
| --- | --- | --- |
| hold | 0.44739903772942186 | 0.448344396857493 |
| forward | 0.4471183361459977 | 0.4477055559518517 |
| left_arc | 0.44590254079608516 | 0.4464097018804202 |
| right_arc | 0.44837465375242913 | 0.44918272124457287 |
| left_turn | 0.44670378695606444 | 0.4474884696098465 |
| right_turn | 0.448260639206121 | 0.44935909453573364 |

The unchanged nominal radius is0.45m. A first-step surface check passing is
not a complete physical-clearance certificate and does not cancel the nominal
veto. Later articulated-surface checks are not provided by this readout.

## Why the existing recovery rule does not activate

The proposal's recorded all-cell`start_clearance` passes at all three inspected
frames:0.4528218653137794m at492,0.4526949885562709m at493and
0.4519749401270365m at503. The observed current pose is still outside the
nominal exclusion radius. The route proposal also has a clear initial connector,
1,503retained floor cells,98occupied cells and a35cell frontier route at493.

`lewm/nominal_clearance_reentry_development.py` activates only for an already
violated current nominal clearance. It returns the original selection when
the current connector is clear. Therefore it correctly stays inactive here:
the problem is forecast feasibility from an admitted current location, rather
than a failure to invoke the existing recovery rule for a violated location.
There are no reentry-candidate receipts at492,493or503.

## Scoring and constraint forecasts differ

The saved prediction arrays already include each model's training-only XY
bias correction. The online residual correction is separate: it adjusts the
first100ms position used for waypoint scoring, while the original nominal
feasibility checks remain on the prediction before that online correction.
This is explicit in`lewm/executed_waypoint_score_development.py`, including
`raw_forecasts_and_constraints_preserved=True` and
`corrected_scoring_path_checked=False` in these receipts.

At503the hold forecast before online correction is
`[-0.006277943029999733,0.004246499389410019]`m. The strictly past-observation
residual correction is`[-0.00636026360310576,0.0042784156241031344]`m, giving
scored hold displacement`[0.00008232057310602735,-0.000031916234693115525]`m.
Thus scoring sees a nearly stationary hold while nominal feasibility uses the
larger uncorrected displacement. No action can win the score after all actions
have been excluded. At492the selected right-turn path had minimum predicted
clearance0.4500224339466468m, only about22micrometres above the nominal radius;
that nominal margin was not a calibrated model-error or physical-safety bound.

These observations do not prove that a residual-corrected action or alternate
plan would be physically safe or complete the maze. They identify a concrete
prospective consistency change to evaluate: use an explicitly declared causal
first-interval correction consistently in candidate scoring and its geometric
checks, while preserving the original forecasts, original veto evidence,
unchanged clearance radius and every remaining horizon's checks. Recheck the
articulated-surface geometry for any changed first-step pose. Do not silently
apply a100ms residual to unobserved longer-horizon errors or feed corrected
predictions back as the raw residual target. Require complete causal prefix,
raw model/command replay and fresh physical execution before adopting a change.

Keep the fixed current cohort and queued baseline comparisons unchanged.
This report changes no controller source, threshold, action menu, checkpoint,
failure status or execution order. The current attempt remains negative and
the broad navigation goal remains unachieved.
