# Nominal requested-motion component for a possible matched prediction comparison

`lewm/requested_twist_forecast_bank_development.py` supplies an explicit nominal
alternative to the learned outcome bank. It accepts only the existing six
ordered eight-interval candidate command plans. It integrates each requested
planar twist over the original 100-ms intervals and returns relative x/y,
sine/cosine yaw and a constant -30 contact logit at all eight horizons.

The integration assumes perfect tracking of requested velocity. It does not
simulate the command limiter, gait, inertia, friction or collision response.
It never updates measured robot pose. The -30 no-event reference follows the
existing zero-reference convention in
`lewm/pulse_timed_training_runner_development.py`; it removes action-dependent
learned contact information and is not a calibrated probability or safety bound.
The bank accepts no observations, future data, geometry or model. Its separate
`nominal_outcomes` key and provenance flags prevent it from claiming a trained
direct or rollout head or a training-fitted translation correction.

Fourteen tests passed in 1.92 seconds, session 80859. They check the analytic
straight-line, circular-arc and turn trajectories, mirrored commands, exact
clock/validity contracts, unchanged inputs, independent outputs, invalid-plan
rejection, explicit provenance and compatibility with the unchanged candidate
cost function. These are synthetic component checks, not navigation evidence.

## Intended comparison and remaining work

A future matched comparison could hold the checkpoint, cost, action vocabulary,
executor, sensing, map, mission and recovery machinery fixed while replacing
the learned outcome bank with this nominal bank at every selection call. The
checkpoint would remain frozen in both arms; its forward contribution would be
disabled in the nominal arm. This would compare learned world-model forecasts
against command-following forecasts under shared downstream control.

It would not be a fully nonpredictive controller: nominal forward integration
still predicts motion, and retained online residual correction would still
adapt from observed execution. Those contributions must be described explicitly.
The existing whole-method reactive arm answers a separate question. Neither
this component nor switching between two heads of a trained model alone proves
the full online-planning causal requirement from handoff section 10.D.

Before any controller comparison, a prospective successor must define the
intervention and its scientific scope; route the provider through every
selection path; preserve shared costs, feasibility, mission and recovery logic;
and correct the original unconditional learned/corrected-model provenance.
Final-goal, waypoint, view-acquisition/reentry, first-interval residual,
hold/anchored recovery and empty-feasible-set branches all need coverage.
The component must not be smuggled into `TrainingTranslationBiasModel` while
leaving `model_prediction_corrected=True` or training-only bias claims attached.

No controller, model assignment, independent-layout treatment, study size or
comparison family was changed. No checkpoints, training data, independent
sensor outcomes or native geometry were accessed by the component tests. The
original eight-diagnostic review and native queue remain pending. This fills in
a concrete candidate forecast source, while the matched controller integration
and physical evidence remain unimplemented and unproven.
