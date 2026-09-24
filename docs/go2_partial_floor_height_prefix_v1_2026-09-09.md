# Partial floor height: prospective command-boundary replay V1

Development only. The completed tracking maze 1 run failed on frame 504 because
82 of 5,956 auxiliary floor candidates exceeded the original 3 mm transported
plane gate. The current plane had enough points but insufficient two-axis
extent. Completed diagnosis d2512d3241e38ac608aa2185cb62e5aafdd4fc13db962fffc68e45d82eaa992e
reproduced that failure. It suggested a scalar height observation under the
existing transported normal; it did not establish physical floor identity.

Candidate: PartialHeightDirectFlowController. Its raw visual tracker, model,
mission, planner, geometry, residual policy and map algorithms retain their
original implementation. Three measured-pose consumers and the registration's
missing-plane transport call bind explicitly to the new typed observation.
Other globals, code, closure and defaults remain original. No imported source
is patched. No performance or hold-policy candidate is combined into this run.

Try original transport first. Only its exact current-candidate conflict permits
the scalar branch, and only with at least 100 candidates and the original
insufficient_combined_two_axis_extent reason. Retain transported normal and
rotation. Shift position along the initial full-floor reference normal by the
negative equal-weight mean signed residual of all current candidates. Require
every corrected point to pass the unchanged 3 mm gate and total correction
within 5 cm; the original rotation correction limit remains 0.10 rad. Reconstruct
all moments and current image/time bindings. Do not trim points, lower extent
thresholds, claim pose-error bounds, rewrite history or promote a partial
observation into a full anchor. Full-plane reacquisition uses the original rule.

Input: go2_direct_flow_maze01_pilot_v1_attempt_001, completed result
d6774bae22cb9effeb0cd85ae255de203de1539541f701d57788b58ab00769de,
case full_jepa_direct_flow_maze_01, original assigned full JEPA model and training
bias. Authenticate completed collection/raw audit/physical prefix, all recorded
artifacts, source closure and original verifiers, plus the exact diagnosis.
Never read sealed material, simulator truth or later outcomes into the controller.

Fresh model/controller/registration/map/mission/history from observation 0.
Replay at most 505 observations, frames 0–504 inclusive. Frames 0–503 require exact
complete saved decisions after normalizing only the successor label and flag,
including model banks, raw tracker, map, mission and actual completed requests.
At 504 require the exact original floor conflict and unchanged raw tracker output.
Validate the live partial pose and retained original anchor. A recovered
controller must produce a fresh learned selection and its corresponding command.
A negative terminal outcome is retained. Stop at 504 whether command changes or
not. Never consume observation 505 or infer the result of an unexecuted command.
Record every returned candidate decision, including comparison failures.

One CPU worker, one OpenCV/PyTorch/BLAS thread, deterministic PyTorch, no training,
unchanged model weights/no gradients, unchanged public arrays. Allow 8 GiB RAM,
512 MiB total output, 256 MiB compressed stream, existing 40 GiB artifact reserve.
Measure CPU topology/affinity/load, RAM, GPU/VRAM, storage and competing jobs
before launch. No native scene is created. CPU analyses may overlap only within
measured headroom. Output root go2_partial_floor_height_prefix_v1_attempt_001
is exclusive; errors retain artifacts and are not retried in place. Recheck
source/input/output bindings at completion. A replay pass authorizes no claim
of navigation or physical safety; fresh native execution and full raw/physical
prefix audit are still required before judging the intervention's outcome.
