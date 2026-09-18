# Comparators with the same measured-plane perception

Implemented `lewm/measured_plane_comparator_controllers_development.py`.
These separate controllers close the estimator mismatch between the prospective
measured-plane learned controller and its comparator methods. Existing frozen
controllers, studies, launchers and running attempts are unchanged.

| Controller / assignment | Forecasts and action selection | Interpretation |
| --- | --- | --- |
| `MeasuredPlaneResidualController` | Assigned learned forecasts, residual correction and predictive feasibility | Existing prospective learned reference |
| `MeasuredPlaneForecastSourceController`, `frozen_world_model` | Same learned reference plus explicit forecast provenance | Common implementation for forecast-source comparison |
| `MeasuredPlaneForecastSourceController`, `nominal_requested_twist` | Eight-step requested-motion integration, constant contact reference, observed residual correction, same scoring and predictive feasibility | Learned versus nominal forecast source; still predictive |
| `MeasuredPlaneReactiveController` | Observed route, current measured geometry and heading rule; no model, residual estimator or candidate future outcomes | Whole-method nonpredictive comparator |

All four use independent instances of `MeasuredPlaneVisualMotion`, the existing
measured-floor registration and persistent map, and the same settled outbound
and return mission. Only the motion observer and explicit result identity are
changed relative to each comparator's existing parent. The parent observation
and control methods are inherited unchanged.

The learned-versus-nominal comparison retains forecasting and does not isolate
whether looking ahead helps. Its nominal assumptions include perfect requested
velocity tracking and no learned contact discrimination. The reactive comparison
changes scoring, feasibility and recovery together; it is not an isolated
prediction-ranking ablation. Neither comparison isolates persistent memory or
JEPA training. Those questions still require separate matched interventions.

Validation command, using the existing deterministic CPU test environment:

```
.generated/venvs/genesis_rocm_0_4_6_v1/bin/python -B -m pytest -q lewm/tests/test_measured_plane_comparator_controllers_development.py
```

Final result: **8 passed in 20.92 seconds**, session 25823, exit 0.
The actual synthetic RGB-D sequence exercises both full-input JEPA-head and
no-RGB direct-head assignments. Each assignment uses three independent
synthetic model fixtures plus the model-free reactive controller. These are
interface fixtures, not admitted trained checkpoints.

Through the first action choice, all arms produce exactly equal visual,
registration, memory and mission receipts, floor/occupied maps and retained pose
history. The first three commands are identical warmup holds. The learned-source
comparator reproduces the reference's complete decisions after removing only
its declared provenance and root metadata. The nominal branch makes zero model
calls while retaining all eight planning segments and observed residual
bookkeeping. The reactive branch has no model or residual object and returns no
forecast or command-integrated pose. Model buffers remain unchanged; duplicate
acquisition stops every arm without extending the map. Inherited-method identity
and invalid forecast-source assignments are also checked.

This is component integration evidence. No new checkpoint inference, physical
episode, command dispatch, prospective timing comparison or independent-layout
outcome was produced. There is no navigation, real-time or hardware qualification.

The queued measured-plane native diagnostic remains the next physical test.
Its outcome must be reviewed before selecting the policy for a population study.
Any population using these comparators needs a separate explicit assignment,
collector/auditor integration, full input admission and prospective execution.
Do not substitute them silently into the frozen original 32-case study. Preserve
all attempts, include failures, match starts and budgets, and stop shared-history
comparisons at the first different executed command.
