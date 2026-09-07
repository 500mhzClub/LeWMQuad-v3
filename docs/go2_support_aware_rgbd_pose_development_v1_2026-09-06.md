# Support-aware reference selection and contaminated-point evidence V1

New fit-only comparison against frozen direct-keyframe V1. No predecessor edits,
new physical collection, validation images, JEPA training, navigation or hardware.
The aim is usable continuous relative-pose evidence, not merely a smaller scalar
allowance. Preserve all earlier stops and failed physical error hypotheses.

## Controlled change

Reuse V1 feature extraction, matching, stable-inlier mean, gyro integration,
six-cell acceptance, motion limits and existing 0.4-m/0.35-rad promotion. Add
promotion AFTER an accepted pose if either reference or current inlier coverage
is <=7 cells: the original six-cell minimum plus one margin cell. Never promote
a failed observation or retry another reference after terminal failure. Preserve
global pose, accumulated anchor error and explicit parent chain. This isolates
reference-selection behavior from a change in feature/registration acceptance.

## Separately conditional outlier evidence

Keep V1's original all-inlier radius as a labeled legacy diagnostic; it is not
validated. Add a separate enclosure around the SAME point-mean pose, assuming
at most floor(0.2*N) of the N accepted pairs are arbitrary outliers. The other
pairs must satisfy the unchanged pixel/depth/gyro hypotheses. The 20% fraction
is a declared development assumption, not a calibrated probability or a fitted
guarantee. Camera calibration remains idealized and gyro-bound failures remain
possible; nothing is widened to cover known outcomes.

Each valid pair gives a ball for the reference-to-current translation with centre
`a - R*b` and radius `point_radius(a)+point_radius(b)+rotation_bound*norm(b)`.
If at least N-f balls contain the true translation, then for each coordinate it
lies above the (N-f)-th smallest lower endpoint and below the (f+1)-th smallest
upper endpoint. Use this outer box with outward arithmetic allowance. It need
not prove a single joint consensus subset. An empty box is an inconsistent
hypothesis and supplies no bound. Around the unchanged mean, take the farthest
box-corner radius and compose it with the retained robust anchor radius and
anchor rotation allowance. If an unknown-bound observation becomes a reference,
its descendants retain unknown global bounds; do not reset them to zero. Fresh
local evidence is not a replacement for missing global history.

## Fixed replay and scoring

Replay the same 336 sustained fitting frames and five V1 members: nominal,
blank RGB, gyro-Z +0.001/-0.001 rad/s, independent depth noise +0.1 mm. Use the
same timestamp-consistent perturbation helper. Predictors see only current/past
RGB/depth/gyro packets. The robust calculation changes no pose or command.
Preserve independent member failures and all not-reinvoked rows. Check exact
pose/orientation/legacy-radius equality with V1 while both still reference frame0.

Save every prediction, failure and keyframe chain BEFORE native scoring. Compare
each member with its SAME-INPUT V1 outcome on common admitted frames; distinguish
that from the older nominal-only ShadowObserver reference. Report completeness,
promotions, position/angle errors, global-radius availability and exceedances,
per-frame point-hypothesis violation counts versus the outlier allowance, local
gyro-rotation hypothesis violations and native translation outside the local box.
An average pose inside its radius does not validate correspondence or gyro
assumptions. No result grants action permission or restores a failed mission.

Before replay, freeze source, protocol, focused tests and narrow import closure
against exact V1 launch/result/prediction/evaluation/analysis identities. Use
exclusive `.generated/go2_support_aware_rgbd_pose_development_v1_attempt_001`.
No overwrite, resume, retry, parameter search or validation-based model selection.
Further source/result auditing must preserve this attempt as recorded.

The longer physical collection, full-body floor coverage, prospective action and
brake validation, common floor/non-floor consumer, timed full maze exploration/
return, matched JEPA training/multistep/memory experiments on independent layouts
and seeds, robustness and bounded hardware remain required for the full goal.
