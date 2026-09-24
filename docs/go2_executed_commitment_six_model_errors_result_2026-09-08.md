# Six fixed models remain inaccurate on executed near-goal commitments

All six fixed family models were freshly admitted and evaluated on the same 46
recorded selection contexts. All 276 candidate forecasts from the original
full-RGB direct controller reproduced exactly. The 276 model/context calls
finished and their predictions/input fingerprints were persisted before native
motion labels were opened. Every model state remained unchanged.

Forty-five complete half-second commitments supplied shared labels, each spanning
exactly 250 physics intervals. The interrupted tick-228 commitment remained
missing. The population contains 41 waypoint commitments, ten within the existing
0.35-m near-goal threshold, 23 action switches, 21 repeats and one initial choice.
Only the actually executed action was labeled; unexecuted candidates were not.

| Fixed model | All XY mean (mm) | All yaw mean (rad) | Near-goal XY mean (mm) | Near-goal yaw mean (rad) |
|---|---:|---:|---:|---:|
| Full RGB / direct | 16.930 | 0.05940 | 21.716 | 0.07420 |
| Full RGB / supervised rollout | 18.690 | 0.05412 | 19.405 | 0.05595 |
| Full RGB / JEPA | 38.925 | 0.09775 | 32.258 | 0.11273 |
| No predictor RGB / direct | 16.082 | 0.02980 | 20.091 | 0.02371 |
| No predictor RGB / supervised rollout | 15.072 | 0.04064 | 15.595 | 0.05484 |
| No predictor RGB / JEPA | 21.285 | 0.05471 | 24.698 | 0.03759 |

For the current full-RGB direct model, XY error averaged 18.929 mm after switches
and 15.354 mm on repeats. This pattern was not universal: no-RGB direct averaged
15.052 mm after switches and 17.193 mm on repeats. The result does not establish
switching as the sole cause. The old data contain moving suffixes of prescribed
plans, which do not cover arbitrary new actions at every moving state.

Concrete current-model errors matter at the 40-mm observed arrival radius:

- Tick 193, repeated left arc: predicted yaw 0.10472 rad, actual 0.22367 rad;
  XY error 16.84 mm.
- Tick 198, left arc to left turn: predicted forward displacement -12.81 mm,
  actual +24.60 mm; XY error 37.87 mm.
- Tick 218, turn to forward: XY error 30.07 mm, including a 24.13-mm lateral
  discrepancy.
- Tick 223, forward to left turn: predicted forward displacement -16.62 mm,
  actual +12.28 mm; XY error 31.52 mm. Hold also had a separate foot-surface
  veto in the native readout.

These are descriptive prediction comparisons on one reused, direct-policy-owned
development trajectory and one optimization seed. They do not establish matched
control outcomes, independent-maze performance, general JEPA/RGB benefit, contact
calibration or the success of another model's unexecuted policy. No model was
selected, trained or promoted. The original visual failure and zero-goal result
remain unchanged.

Two focused coordinate/validation tests passed in 1.78 seconds. Preflight recorded
82.15 GB available RAM, 84.73 GB artifact storage free, 0.3% CPU utilization and
idle GPUs. The small fixed-model inference workload ran sequentially with one
thread. Post-launch work took 16.631 seconds. The result binds 1,030 source files
and authenticates all native/readout/fit artifacts before and after.

Artifacts in `go2_executed_commitment_six_model_errors_v1_attempt_001`:

| Artifact | SHA-256 |
|---|---|
| `launch.json` | `c4de58bf409aaaaa0e6574b51912d3ddb95b9e1378cb9f5eb5f4e0f76a45b91e` |
| `prediction_phase_complete.json` | `b85a6f3f59713e16c373b4f9c69b89305debb732ca97d76494847565a6c97580` |
| `evaluation.json` | `3437a89e9103751738e7213216fae2f2ea40212c33379aa103f6e81f46095247` |
| `result.json` | `6d3ce4bf199e9ca464c24d7a0b8e7ba19c8392198d58ac0f0152ff6b6a20065e` |

Next, implement a bounded prospective moving-action collection covering every
prefix/suffix pair, including repeats and braking, with exact physical/context
prefix matching across alternatives. Keep raw measurement, contact censoring,
geometry-role separation and failures intact. Assess source correctness and
actual useful collection concurrency before scientific collection. New fitting
and matched closed-loop execution must follow complete data admission; none of
these prediction scores completes the end-to-end navigation goal.
