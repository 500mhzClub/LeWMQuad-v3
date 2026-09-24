# Ground-veto attribution: uncertainty is not a measured wall

The recorded lower-leg vetoes have now been traced to their individual depth
views and error terms. Every one of the 12 veto witnesses at each of the 0.75-m
and 1.00-m tangent configurations disappears in the diagnostic branch that
recognizes the independently observed plane family. All original query results
remain unchanged. This is an explanation of existing negatives, not permission
to execute either configuration or a new maze result. Full mission remains 0/2.

## What the new evidence establishes

The source-bound replay reproduces all 12 startup decisions and 15 relative
observations. At each distance it reconstructs 27 primitives in five views:
135 whole-query witnesses, 270 total. Both beam branches match their reference
implementations, the complete original queries match reference, and the exact
per-shape obstacle/penetration timestamp lists are independently reproduced.

The four investigated shapes are FL/FR `calflower:0` and `calflower1:0`.
Their nominal physical minimum floor gaps are approximately 27–54 mm. In
historical views, relative point allowances increase from approximately 33 mm
at 1.5 s to 41 mm at 1.7 s, 52 mm at 1.9 s and 65 mm at 2.1 s. These are
uncalibrated inherited pose proxies, not measured obstacle heights. The smaller
lower-leg primitives lose definite floor separation in all four historical
views; the larger two lose it in the last two historical views. The original
consumer then disables its plane-family exemption and records near floor-like
returns as obstacle vetoes: 2+2+4+4 = 12 per configuration.

At 1.00 m, the current 2.9-s view cancels common pose uncertainty and yields:

| Physical primitive | Minimum floor-gap lower bound | Observed floor coverage |
| --- | ---: | --- |
| FL calflower | 49.591 mm | complete |
| FL calflower1 | 26.040 mm | complete |
| FR calflower | 52.004 mm | complete |
| FR calflower1 | 29.181 mm | complete |

For all four shapes at 1.00 m, all five diagnostic plane-masked whole-box queries
are clear under the supplied model. This does not establish clearance for the
other 23 primitives, their required residual boxes, support, or any intervening
trajectory. Original conditional counts remain 20/27 at 0.75 m and 19/27 at
1.00 m, with the same four lower-leg vetoes and no penetration flags.

At 0.75 m, none of the four plane-masked whole-box queries is complete in any
view. Only the current FL calflower1 footprint has complete floor coverage.
Thus separating the semantic channels cannot fill the missing visibility.
The observed plane family itself remains conditional on its declared errors;
classification is not an independently calibrated ground-truth label.

## Required implementation consequence

Do not globally drop old observations or shrink their errors. Instead, the next
consumer must report three independent facts for the same supplied configuration:

1. Non-floor return conflict, retaining a veto from any view after classification
   against that view's bound observed plane family. If no valid plane exists,
   do not invent one or suppress the raw conflict.
2. Physical ground relation: separated, intersection possible, or penetration
   under every supplied model, with per-view intervals and actual coverage.
3. Positive complete-box/residual visibility, separate from absence of conflict.

An interval straddling zero is absence of proof of separation, not proof of
collision. It must not overwrite independent, compatible, better-constrained
evidence of separation merely because it is older. Conversely, genuine
contradictory measured obstacles, incompatible plane hypotheses, certain
penetration and unobserved required volume must stay unresolved/rejected. Any
cross-view combination must explicitly check frame/shape/plane correspondence;
do not average away disagreement or turn a union of disjoint observations into
a filled volume. Exact foot geometry remains a contact candidate, not a licence
for lower-leg contact or a friction/support model.

The new diagnostic is deliberately not installed as an online controller. Its
unconditional plane branch is an attribution intervention. The unchanged
consumer continues to deny the original configurations. This avoids silently
changing an acceptance rule in a completed experiment.

## Next action toward execution, not more static offsets

Implement the factored evidence interface together with an action-conditioned
short-horizon trajectory interface: current body/joint state, applied-command
history and candidate command sequence in; predicted body pose, joint/foot
configurations, stopping trajectory, model provenance and error terms out.
Constant-posture translations are only an explicitly labelled baseline. A
requested velocity is not an observed displacement or a certified motion bound.

Use a fresh bounded development execution to identify and test motion errors,
starting from the verified observation maneuver and preserving the same state
owner throughout. It needs its own declared initial-region duration, live
admission, source/input/native bindings, command schedule, speed/contact guards,
complete stopping tail and raw replay auditor. Never extend the old 3.5-s expiry
or resume its recording. Include forward initiation, sustained translation,
turning and braking; evaluate body and articulated geometry throughout each
horizon, including between-sample motion. Split identification and validation
executions, retaining failed trials and distinguishing empirical coverage from
a guarantee. Runtime inputs must remain deployment-valid; native pose/contact
truth is evaluation-only. Record complete-loop latency rather than assuming
simulation-time 10 Hz means real-time operation.

Then connect the action/evidence consumer to a fresh continuous discovery,
marker and return mission. The subsequent matched supervised/JEPA/geometry,
one-step/genuine-multistep, memory, independent-layout/seed/robustness and bounded
hardware requirements in the continuous-navigation plan remain outstanding.

## Verification and retained artifacts

- Focused session 51759: four tests passed in 5.63 s, including a synthetic wall
  whose conflict survives the plane branch, exact original-witness reproduction,
  same-frame cancellation, out-of-view queries and stale-state rejection.
- Diagnostic 76444: exit 0; stdout exceeded the tool retention limit. It is not
  used as the retained complete result. Only stdout selection was subsequently
  shortened, after the process was terminal; no evidence rule changed.
- Diagnostic 13743: exit 0, bounded report retained in
  [the machine-readable result](go2_configuration_ground_veto_diagnostic_result_2026-09-06.json).
  All 270 witnesses were checked; output retains all vetoes plus all five views
  for the four investigated shapes, including non-veto witnesses.
- Expanded regression 75366: **1,749 tests across 142 explicit files passed**
  in 120.13 s. No tested source changed during the run; all handles are terminal.

The diagnostic verifies the three fixed launch/result/audit identities,
385-source/inherited-input/74-artifact and 16 native bindings before and after,
plus 14 explicit development/diagnostic paths. Exact hashes are in the retained
result. No frozen runtime source/input/protocol/output was changed, no sealed
material was accessed, and no physics, training or navigation command ran.

Retained result SHA-256:
`bd2d9b65b7517f8b12c0906f23346e5824c735941e44de7ee9d76412c8eae449`.
