# Geometry-dependent local progress pilot V1

Prospective training-only development collection. This follows the failed RGB
benefit diagnosis in the independent pulse study and the unpaired joint room
return result. It does not rerun either experiment or qualify navigation.

Freeze this document, new source, focused tests and inherited dependencies in
the launch receipt before the first native episode. Use only the fresh root
`go2_geometry_progress_pilot_v1_attempt_001` under the existing external
navigation-development artifact base. One collection and one raw audit; retain
all failures. No retry, resume, action/geometry tuning within the attempt,
checkpoint fitting, evaluation-set access or hardware execution.

The cohort has 24 episodes: two mirrored partial front panels, two appearance
seeds and six actions, all fully crossed. The panel is at x=0.65m, thickness
0.08m, height 0.70m and y extent [-0.95,0.05] or [-0.05,0.95]. Room boundaries
are fixed. Robot starts at (0,0), nominal friction 1.0, existing checkpoint gait,
CPU simulation, ideal body/gyro and hidden-robot RGB-D. Three quiet 100ms ticks
after the unchanged 1.5s settling establish four current/past RGB/body packets.

Action horizon: 40 ticks (4 seconds), including ten terminal zero ticks.
Hold: all zero. Forward: vx=0.20m/s for 30 ticks. Left/right arc:
vx=0.16m/s, yaw=+/-0.45rad/s for 20 ticks, then vx=0.16m/s and zero yaw for ten.
Left/right pure turn: zero vx, yaw=+/-0.45rad/s for 30 ticks. Lateral requests
remain zero. These are command prefixes, not assumed motion. All eight blocks
use the existing JEPA action normalization and are known at departure.

Evaluation-only target is (1.2,0)m in the departure body frame. Successful local
progress requires the full actual horizon, no disallowed native contact or
other physical/acquisition stop, and at least 0.15m reduction of distance to
that fixed target. This does not mean passage traversal. Every actual contact,
fall, speed/domain stop, setup rejection, missing image or truncated horizon is
retained. Noncontact stops cannot prove future safety. Contact-positive labels
survive missing future images; missing terminal motion is never synthesized.

Informative-design gate, fixed before data: both appearance seeds must show
successful left arc and failed right arc for the left-open panel, and the
reverse for the right-open panel. Hold and pure turns must fail progress in
all four strata. No constant action may succeed across all four strata.
Failure of this gate diagnoses the design and blocks scaling this unchanged
panel into a larger learned comparison. Native outcomes, not a kinematic proxy,
decide the gate. No prediction training is part of this pilot.

The fixed seeded shuffle assigns opaque episode IDs independently of geometry,
appearance and command labels. Scene construction receives no candidate action.
These are independent episodes in balanced geometry/appearance strata, not
same-image counterfactuals. The audit records native/history/RGB equality and
differences across sibling episodes without excluding unequal prefixes. There
are zero independent maze evaluation layouts and no statistical RGB/JEPA
benefit claim from 24 development episodes. Future learned-policy comparisons
need independent layouts, matched randomized trials and ablations.

Retain raw physics, attributed contact, actual requested/applied command tapes,
all acquired RGB/native-depth/public-depth, body/control/gyro histories, native
geometry/setup evidence, friction, gains, and per-decision wall time. Use the
existing raw sensor/contact/depth reconstruction and articulated setup/stop
checks. The audit replays every sensor-valid action decision and shadow observer,
checks actual command-prefix identity, and scores all 24 episodes. Frozen old
modules are imported or separately adapted; none is edited or monkeypatched.

Resource admission: at least 24GiB available RAM; 3GiB planned storage plus a
40GiB reserve; stop on reserve breach. Serial native scene, one OpenCV/BLAS
thread; no concurrent heavy job. CPU/RAM inventory is captured at preflight.
Physics pauses during computation. Acquisition/control wall times are recorded,
but this fixed-tape experiment does not repair the previous real-time deficit.
No deployment, real-time, maze-navigation, RGB benefit or goal-completion claim.
