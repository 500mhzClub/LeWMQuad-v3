# Measured visual continuation from an admitted floor pose

The eleventh episode stops at1904 while current auxiliary visual tracking is
valid. Fixed-population diagnosis1677bd7be89e5dd85c2e467693575aa5144f3f72fcef09ab16e8afa38b513a35
reconstructs the unchanged extent rejection. This successor tests whether a
current measured pose can continue without an admitted current floor plane.
It changes the pose hypothesis during floor missingness; it does not establish
an uncertainty bound or resolve the separate strict visibility failure.

Retain the most recent fully admitted floor pose (a,A) and its original visual
pose (b,B). With current independently admitted visual pose (p,R), compute
a+A B^T(p-b), A B^T R. This uses current visual measurements. Do not use command
integration, native poses, gyro extrapolation, historical-map rewrites or an
unadmitted floor pose as an anchor. Current point count/extent missingness is
the only entry path. Available planes use the exact original registration;
up/coherence/correction conflicts retain terminal failure behavior.

Represent transport in a separate evidence schema and accessor. Preserve the
original unavailable current-plane receipt, both current camera hashes, the
full admitted anchor and its raw visual witness. Reconstruct the missingness
from original untrimmed candidates. Check every candidate against the initial
floor plane transformed by the transported pose, with the original3mm limit.
No points means no geometric agreement, not a zero residual. Preserve the
original5cm/.10rad correction magnitudes as development rejection gates;
they are not pose error bounds. The new position gate measures full correction
norm, and the rotation gate measures full correction angle. There is no newly
invented temporal uncertainty guarantee: anchor age is explicit, current raw
visual admission (including its existing10-frame measured-bridge limit) stays
mandatory, and the original mission/history limits remain. Floor reacquisition
uses the original registration and replaces only the most recent floor anchor.

All pose consumers use explicit schema dispatch; old evidence retains its
unchanged accessor. Mapping, contact queries, residuals and mission positions
share the same current pose. Settling receipt wording now says admitted visual
positions in the floor reference, because a current plane may be unavailable.
The learned model, map implementation, selector and settling rules stay fixed.
The separately tested map-performance candidate is not included here.

Run scripts/replay_go2_measured_floor_transport_prefix_v1.py with the actual
completed eleventh --native-result-sha256 and --preflight-only. Require all raw
sensor/model/actual-command audits and its physical/public/prospective prefix.
Require all65 diagnosis input bindings to match the final native artifact map.
Bind complete inputs, runtime predecessor admission, source closure and model.
One CPU/numerical/OpenCV thread,8GiB available RAM,1GiB output above40GiB reserve,
no scene or training. Independent analysis may overlap the bounded renderer
probe if actual resources permit; do not create a second native scene.

Replay fresh complete controller state from observation0 through1904. For all
1904 earlier observations, compare the complete decision exactly after only
validated controller-label/feature-flag and settling-source-wording normalization;
all actual prior commands and public input arrays must match. At1904 require
the same raw current visual evidence as the recorded failed controller, an
explicit valid transported pose from exactly the recorded1903 admitted anchor,
active RETURN mission and no terminal failure. Preserve the original failed
decision beside the intervention. Stop there: do not consume later recorded
observations or infer outcomes under the changed command.

Exclusive output go2_measured_floor_transport_prefix_v1_attempt_001. Preserve
all failure evidence. Completion only admits preparation of a fresh prospective
navigation experiment with its own physical/raw audit; this replay proves no
return, independent layout performance, timing deadline or deployment claim.
