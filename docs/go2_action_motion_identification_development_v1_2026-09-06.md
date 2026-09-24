# One fresh action-motion identification acquisition A

This fixed development execution instantiates run A of
`docs/go2_action_conditioned_motion_validation_next_execution_2026-09-06.md`.
Output: `.generated/go2_action_motion_identification_development_v1_attempt_001`.
Scene: `go2-action-motion-identification-development-v1`; seed 2026090603.
No retry, resume, validation-run entry point, sealed access, hardware actuation,
JEPA training or maze-success claim. Do not launch B before a model from A and
its reporting rules are frozen. Failed or partial A remains a recorded result.

Reuse the bounded four-wall scene, aligned physical/visual floor, learned gait,
checkpoint, gains, body/fast-gyro/RGB-D sensors and native readback identities.
Run CPU physics with software EGL. After exactly 1.5 s of settling independently
admit a NEW initial-body non-floor region [-1.25,1.25]^3 m, static through 8.0 s,
and initial velocity 0 +/- .02 m/s. Retain the original instantaneous check as
a witness, separately reconstruct the larger condition, and bind its admission
to `motion_setup_checks.json`. This is a calibration-arena assumption, not a
depth measurement, an extension of the old recording, or a deployment maze prior.

One MotionState owns the original startup controller, relative observer and
memory through the completed observation turn and three measured zero-tail
frames. The old terminal startup controller is never restarted. Only after
READY does the new factored continuation guard take over. It retains measured
non-floor conflicts, covered penetration and incompatible covered ground planes;
missing ground visibility remains unknown, not support permission. The old
current-primitive classification remains a labelled comparator, not silently
removed evidence or the new guard. A failed owner cannot resume.

After READY execute 28 targets at 100-ms intervals: six [.12,0,0], four zero,
four [0,0,.35], four zero, six [.15,0,0], four zero. At every prospective motion
decision require the checked starting region to contain the unchanged all-joint
radius plus .04-m padding, .3-m/s base-speed allowance for .4 s, and the current
fusion position allowance through the entire horizon. Require inferred speed
plus its prior contribution <= .3 m/s. The final schedule boundary needs inferred
speed <= .05 m/s and all 50 new fast-gyro samples <= .1 rad/s. Then acquire
three more real zero-command ticks with continued observation when healthy.
No physics follows a native physical stop. A returned observer/controller
failure requests the real zero tail when the physical session remains healthy;
an unexpected raised contract/infrastructure error terminates and retains its
partial trace, with missing stopping evidence reported rather than fabricated.

Native evaluation guards run at 500 Hz for the inherited body/contact checks,
exact non-foot ground contact, .3-m/s actual base speed and padded articulated
body containment in the new region. Native pose/contact truth never enters the
policy or prediction inputs. Full-loop timing is recorded; simulation-time
10 Hz is not a wall-clock real-time claim.

Every scheduled motion decision saves a four-step ideal-command/gravity-tangent
and measured-joint-velocity persistence prediction using only its current causal
packet and the next declared targets (zero after schedule end). The forecast is
NOT a learned gait, calibrated error bound or stopping guarantee. Do not use
the earlier interface-only .05-m / 2-m/s assumptions for command acceptance.

The independent auditor reconstructs actual sensor histories, startup and
factored continuation, decisions, predicted inputs, command slew/timing, new
setup, foot identities, contacts, actual stopping and sampled prospective
envelopes. Score each 0.1/0.2/0.3/0.4-s prediction only if its horizon is fully
recorded AND its predicted applied-command sequence was actually executed.
Report truncated and changed-command cases explicitly. Native endpoint errors
are evaluation-only. Primitive material-point displacement upper bounds use
centre displacement plus rotation-matrix spectral difference times each shape's
circumsphere radius. Sample-to-sample displacement rates are NOT continuous
physical speed bounds. Report original baseline errors before any fit.

Before launching, bind the explicit new recursive source inventory together
with the frozen startup source/input/artifact/native identities and the exact
final factored-interface result/source witnesses. No whole-tree materialization.
Run focused and explicit expanded tests before physics. Freeze launched sources,
inputs, protocol and recordings. Audit independently after acquisition; no
intermediate test, local run or successful audit completes the navigation goal.
