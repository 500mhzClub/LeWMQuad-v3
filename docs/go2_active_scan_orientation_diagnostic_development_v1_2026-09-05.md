# Active-scan orientation error diagnostic

Posthoc-motivated diagnosis, specified during the original fixed scan panel
after completed wider scans showed approximately 0.045–0.049 rad maximum
gyro-heading discrepancy. This does not change its source, commands, outcomes,
0.12 rad physical heading criterion, or its sixteen specimens.

After the original sixteen scans complete and pass raw audit, replay every
available control decision of every specimen, including truncated contact
failures. Compare three fixed integrations: existing midpoint exponential,
midpoint plus the first linearly varying-rate coning term
`cross(omega_previous, omega_current) * dt**2 / 12`, and right-endpoint
exponential. No coefficient fitting, selection from a sweep or physical rerun.

Two data rates are kept categorically separate: actual recorded 50 Hz body gyro
samples versus a 500 Hz EVALUATION-ONLY diagnostic formed from recorded native
world angular velocity and true rotation. The latter is unavailable to the
deployed packet interface. It diagnoses sampling/timestamp/numerical limits;
it is not additional observed IMU data or permission to feed privileged physics
into a policy. No denoising, invented intermediate measurements or future samples.
All methods start at the same post-settle identity. Evaluate only the original
controller's actual decision timestamps; do not extrapolate past native contact.

Exact live 50 Hz midpoint rotation replay must agree within 1e-12. Report all
six method/rate combinations, full SO(3) and projected heading errors, terminal
signed errors and per-specimen counts. Group identical raw physical arrays and
report their repeated geometry membership; sixteen rendered specimens need not
be sixteen independent physical trajectories. No confidence intervals from
correlated repetitions, and no replacement scan-success score.

If 500 Hz improves error, a future deployment-rate sensor intervention requires
new actual causal capture and explicit sampling/latency contracts. If the
coning term helps at 50 Hz, any controller change still requires separately
specified physical validation. Neither result fixes calf-wall collisions.

Exact root: `.generated/go2_active_scan_orientation_diagnostic_development_v1_attempt_001`.
