# Commitment-horizon waypoint pose utility V1 — source component

The matched-model native cohort remains unchanged. This separately named source
component is prepared while that cohort runs; it has no execution root, selected
model, training, simulator or hardware authority of its own.

Displacement-only waypoint scoring gives an in-place turn no explicit value for
orienting toward a sideways target. Moreover, its four-second contact cost can
penalize motion beyond the next five committed 100-ms commands. The proposed
revision evaluates each model's first half-second predicted XY and yaw using
the waypoint potential

`V(g, yaw) = ||g|| + 0.4 * min(||g|| / 0.35, 1) * (1 - cos(atan2(g_y,g_x) - yaw))`.

At zero distance the alignment term is zero. Utility is initial potential minus
predicted potential minus `1.2 * predicted_half_second_cumulative_contact`.
The target vector after motion is `g - predicted_xy`, with heading measured
relative to the current body frame. The alignment term fades continuously near
the target. The 0.4-m turning scale and 0.35-m fade distance are fixed development
choices, using the existing scan scale and waypoint lookahead magnitude; they
are not fitted physical constants or calibrated risk bounds.

Retain all six complete forecasts, original utilities and the new distance,
alignment and contact contributions. Ties follow the original action order.
An undefined yaw rejects the component call. The caller must still apply the
same measured articulated-surface vetoes. No measured obstacle conflict is
overridden, no unknown space becomes certified free, and no model prediction
is corrected. The scan utility itself is unchanged. This alters both the
waypoint value and its scoring horizon and must be described as such in any
later prospective experiment, rather than as an isolated heading ablation.

Five synthetic test cases cover left/right alignment, forward progress, the
committed contact horizon, preservation of original forecasts, a finite vanishing
alignment term near the target, and undefined-yaw rejection. Tests and any
native integration follow the current cohort to avoid adding concurrent compute
to its recorded loop timings. Successful component tests would not demonstrate
goal-reaching or authorize modifying the running/frozen comparison.
