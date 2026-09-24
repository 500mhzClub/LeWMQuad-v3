# Course-aware controller source and supervised response diagnostic

Use only already exposed V1 visual-servo sensor/controller records. No native
pose or friction label enters features or targets. No new physics in this fit.
Freeze the new course estimator, controller source and tests with this launch;
do not retune them after seeing the diagnostic. A later physical collector must
have its own fixed starting conditions, protocol and exclusive output.

Course: six exact 10Hz visual poses; least-squares velocity over the past 0.5s,
unwrapped mean yaw/rate. Below 0.012m/s, course is explicitly unavailable.
Current-course hypothesis transports the measured course offset from mean body
yaw to current yaw, assuming slowly varying slip. Residual scatter is not a
physical bound. The new controller corrects target-bearing minus this course
hypothesis, with the old body-heading assumption explicitly retained only when
course is unavailable. No native sideslip or simulator friction coefficient is
used. Forward gain 1.8, cap 0.12m/s, cosine steering reduction, no persistent
minimum speed. Yaw gain/cap 1.5/0.25rad/s, same complete 0.4m/+0.3rad target and
arrival/settling/excursion rules as V1. New prospective control timeout 35s
accounts for observed slow translation and heading correction; V1 stays failed.

Regression inputs: [1, proposed vx, proposed wz, vx*wz, past visual vx_body,
past visual vy_body, past visual yaw rate]. Targets: next 100ms visual translation
in current body axes divided by dt, and next visual yaw change divided by dt.
Only complete past windows and available subsequent labels enter fitting; no
future label enters the feature window. Standardize by fitting-set RMS and solve
ridge 0.01 with an unpenalized intercept. Report matrix rank/singular values,
in-sample and both leave-one-condition-out errors, against past-motion persistence
and command integration. These correlated closed-loop trials are exposed
development data, not independent validation or identified causal action effects.
The fitted coefficients are NOT used by the new controller in this experiment.

This is a small supervised baseline, not JEPA or proof that adding action inputs
helps. No calibrated error bound, controller promotion or full-goal achievement.
Exclusive output: `.generated/go2_visual_course_response_v1_attempt_001`.
