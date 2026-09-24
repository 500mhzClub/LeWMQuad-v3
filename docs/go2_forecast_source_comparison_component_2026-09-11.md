# Matched learned-versus-nominal forecast intervention

The controller implementation now has an explicit forecast-source intervention.
It is not assigned to the fixed independent study and has no recorded-data or
native execution result. It addresses the missing operational definition noted
in `go2_goal_causal_comparison_completion_gap_2026-09-11.md` without declaring
that the broader causal requirement is complete.

## Operational definition and scope

Both modes use the same `ForecastSourceResidualController`, observation and
mission code, map, residual update rule, six ordered actions, eight 100-ms
prediction slots, cost functions, phase allowances, footprint checks and
recovery functions. A future matched comparison must load separate copies of
the same admitted frozen checkpoint and correction buffers and bind identical
sensing, geometry, execution budgets and initialization.

- `frozen_world_model` invokes the original corrected prediction selector.
  Its outputs enter the original costs and feasibility rules. The new receipt
  records that observation history was supplied, not that a learned visual
  dependence or navigation benefit has been demonstrated.
- `nominal_requested_twist` retains the assigned model but does not call its
  forward method. It substitutes the already implemented exact planar
  integration of requested candidate twists, assuming perfect velocity
  tracking. Contact uses the fixed -30 reference logit. No learned translation
  bias is applied. The head and correction flags explicitly identify this
  nominal source, and its assumptions accompany the forecast.

The latter remains predictive: nominal kinematics and the shared observed
residual correction still influence planning. This is a comparison of learned
forecast use against an adaptive nominal reference, not a fully nonpredictive
controller or a switch that disables every form of lookahead. The existing
whole-method reactive baseline remains a separate comparison. Effects of JEPA
training, planning horizon and persistent map information require their own
matched evidence. No causality or navigation result follows from this component.

## Implementation

`lewm/forecast_source_selection_development.py` implements the provider choice
under the original candidate-input and cost interfaces. It requires the same
evaluation-only corrected-model interface in both modes but does not admit a
checkpoint. Nominal forecast values do not depend numerically on model outputs,
observation-history values or learned XY biases; causal history validation is
retained. The controller's other sensing paths still use their observations.

`lewm/forecast_source_residual_controller_development.py` reuses the original
waypoint method with a private provider binding. Its cooperative inheritance
retains final-goal scoring, view reentry, intermediate-waypoint execution
scoring, first-interval correction and anchored hold recovery in their original
order. The eight-step layer preserves the provider's correction flags across
the original scoring functions, which otherwise overwrite that historical
metadata. Costs, forecast paths and recovery arithmetic are unchanged.

The original module globals, controller sources, model adapters, running
native jobs and fixed independent-study definition are unchanged. Constructor
selection is explicit; the public forecast-source property is read-only.

## Validation and remaining work

The final focused run passed all 49 tests in 10.83 s (session 13460), covering
`test_forecast_source_selection_development.py` and
`test_forecast_source_residual_controller_development.py`.

The tests compare complete learned-mode selections with the original selector
for intermediate targets, final goals, view acquisition and blocked actions.
They exercise nominal forecasts through those same functions, positive
translating reentry, positive anchored hold recovery, exhausted view search,
unchanged residual targets and latched sensor failures. Four public RGB-D
packets with explicitly synthetic dual-camera pose witnesses reach actual
mapping, forecast selection and residual bookkeeping in both modes. This test
does not validate a visual tracker or physical execution.

A small CPU neural-model test checks unchanged parameter/buffer bytes, absent
gradients and unchanged input tensors across both provider modes. Synthetic
forecasts also demonstrate that the source intervention can change the selected
action under the same cost. Neither test uses an admitted trained checkpoint
or establishes a learned advantage.

Initial tests caught a correction-flag restoration bug, which was fixed before
this component was accepted. The four-packet fixture was extended with explicit
dual-camera bindings and constant floor height; its predecessor deliberately
changes height and correctly crosses the original correction gate by frame
three. No sensing gate was relaxed. The initial neural test also emitted a
Torch RNG device-initialization warning; the final test explicitly restricts
RNG preservation to CPU with `devices=[]`.

Still required before scientific use: exact assigned-checkpoint/factory
admission, a prospective study definition and analysis that state the above
contrast, and matched closed-loop physical simulation executions. The pending
native diagnostic review must determine the controller and budgets for that
study. Do not insert an extra arm into the original 32-case verifier or claim
that recorded-tape rescoring demonstrates navigation benefit. The active
single-pass performance replay and extended-budget navigation audit retain
their original scope.
