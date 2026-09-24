# Instantaneous articulated scan geometry diagnostic

Specified after the completed, raw-audited sixteen-scene active scan and eight
paired-rate scans. This is a new sensor-geometry component and a posthoc contact
diagnosis, not a simulator/controller retry or a clearance qualification.

Implement all27 box/cylinder/sphere collision primitives in the SHA-bound Go2
URDF, with full fixed/revolute joint chains and collision origins. In particular,
the two fixed calf child cylinders remain part of the corresponding collapsed
calf rigid group, while separately retained feet remain distinct. Resolve nominal
groups against actual native topology; do not assume the calf has only one shape.

Runtime inputs are the existing strict actual RGB/body packet, its current valid
twelve ordered joint positions, and the fixed robot calibration asset. No world
pose, walls, contact labels or future postures enter runtime body-extent outputs.
Compute exact primitive support along body X/Y/Z and the union's support bounds.
This is an instantaneous nominal outline; support bounds also bound the convex
hull and do not describe gaps between links. Joint uncertainty, future gait
sweep and environment clearance are explicitly unqualified.

Replay all4,504 actual controller decisions from all24 recorded scan specimens,
including all failed/contact trajectories. At every frame compare sphere-center
foot kinematics with the independent closed-form implementation. Preserve all
body bounds, not only a selected favorable pose.

Separately inspect all native disallowed contacts in the six contact trials.
For each identified wall, evaluate the inward-facing wall plane along the wall's
thin axis and compare true current-pose projections of torso-only support, whole
robot support, and the reported native rigid group's support. Report plane gaps,
limiting primitives and contact-position plane residuals. This calculation uses
true pose, contact identity and wall geometry ONLY for evaluation. It is an
infinite-plane diagnostic, not a finite-wall intersection solver or an independent
reproduction of contact forces. Contact-selected examples do not calibrate a
predictive safety classifier, and a positive/negative plane gap is not a future
turn certificate.

No coefficient, inflation margin, gait, control rule or learned model is fitted
or changed. All source, URDF and exact predecessor artifacts are bound before
analysis and verified afterward. Keep full old physics/decisions/results intact.
This should guide the next observation-to-traversal integration step and a
separately tested future-motion envelope, not another unmotivated model sweep.

Exact fresh root: `.generated/go2_articulated_scan_geometry_development_v1_attempt_001`.
