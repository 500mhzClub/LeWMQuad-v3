# Orientation diagnosis: sampling dominates the tested numerical correction

The corrected fixed diagnostic completed all sixteen recorded scans and all
six integration/rate combinations. It changes no physical execution or scan
success result. All3,040 original decision rotations replay exactly with the
50 Hz midpoint baseline (maximum matrix difference0).

The two full-scan physical trajectories have the following maximum projected
heading errors over their actual decision frames. Each repeats across six
different scene specimens; these are two trajectories, not twelve independent
accuracy samples.

| Rate and integration | Initial heading−0.15 | Initial heading+0.15 | Input status |
|---|---:|---:|---|
|50 Hz midpoint |0.048811 rad|0.045008 rad|Actual recorded policy gyro|
|50 Hz coning correction |0.047852|0.044030|Same recorded gyro|
|50 Hz right endpoint |0.051106|0.047180|Same recorded gyro|
|500 Hz midpoint |0.000404|0.000400|Privileged evaluation-only rates|
|500 Hz coning correction |0.000407|0.000404|Privileged evaluation-only rates|
|500 Hz right endpoint |0.00000666|0.00000670|Privileged evaluation-only rates|

The coning correction was fixed analytically, not fitted. It is much too small
to explain or remove the observed50 Hz accumulated error. Changing to a
right-endpoint rule at50 Hz worsens maximum and mean absolute heading error,
even though its final signed error is slightly smaller. Selecting only that
last value would conceal the tradeoff.

Increasing rate in this diagnostic reduces the error by roughly two orders of
magnitude even with the unchanged midpoint integration. This identifies a
sampling/discrete-time limitation in the current ideal-sensor pipeline, not
evidence of hardware gyro bias or a need for a larger learned world model.
The near-exact500 Hz right-endpoint result is consistent with matching the
simulator's discrete update convention; it does not establish superiority for
an actual IMU. Real measurements have sampling, filtering, delay, noise, bias,
mount and clock characteristics not represented by these ideal rates.

The500 Hz values were explicitly reconstructed using saved native world angular
velocity and true body rotation. They are diagnostic references, not deployed
measurements, interpolated50 Hz observations or new policy inputs. A future
controller must obtain genuinely causal high-rate virtual/real IMU samples or
preintegrated sensor deltas through a separately tested interface. It cannot
reuse this privileged replay as its runtime state.

## Complete population and integrity

All sixteen specimens contribute, including the four truncated contact scans.
There are four exact raw-physical-array identity groups: two contact trajectories
repeated twice each, and two completed trajectories repeated six times each.
Grouping binds array names, shape, dtype and bytes. This confirms the original
scan report's warning against treating these as sixteen independent physical
replicates. The rendered scene observations remain different scientific inputs.

V1 failed before integrating any trajectory because `array_binding` returns a
dictionary, which was incorrectly used as a dictionary key. Its source, empty
FAIL result and launch remain unchanged. V2 corrects only that accounting
operation, binds the predecessor failure, and passes thirteen focused numerical
and accounting tests plus an actual223-decision baseline preflight. The full
scan's original raw audit remains PASS. No simulation or task outcome was rerun.

The corrected diagnostic checks all source/input bindings before and after,
uses the raw-audited scan population, and includes exact baseline replay. Its
alternative-method reductions are tested calculations, not an independently
implemented second auditor or real-platform qualification.

Exact root: `.generated/go2_active_scan_orientation_diagnostic_development_v2_attempt_001`.

- Corrected launch: `42809131032f97f428b80fad3ff1334ff2990a04d76fff4d8d66a675e86d9332`.
- Corrected result: `391e8471eb0325928a625d7586d452e014648fa3be64e94207b1ea5be686710e`.
- Preserved V1 launch: `0bbb0a612f655f4b5386db74596d9d43013fa2f3b375c4a707e556124b0e3437`.
- Preserved V1 FAIL: `334df1fa095b62a76b4d9bd5bb07274a15e867e185ffc52b065f7843927f0005`.

Immediate next work is the [sensor-to-navigation integration plan](go2_scan_to_navigation_next_steps_2026-09-05.md),
not an in-place change to the frozen scan or another unmotivated training sweep.
