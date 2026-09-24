# Direct-flow observer: fixed full-controller maze1 prefix V1

Development-only prospective command replay, not a native navigation result.
Use the completed independent learned cohort result
`a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720`,
maze1 and its assigned full-JEPA model with unchanged training-only correction.
Reconstruct paired public observations from episode start through observation
214 inclusive. The original controller first fails at that observation, before
advancing its decision tick from 213. Never consume observation 215, even if the
new command remains zero. No future recorded consequence belongs to a changed
command or terminal policy.

The new observer tries the complete original two-camera policy first. Only its
two missing-pose statuses permit a second pass. Direct corner optical flow can
replace a failed original correspondence set only against the immediately prior
100 ms reference. Original rigid geometry, gyro agreement, reference conflict,
anchor/increment continuity and ten-frame measured bridge limits still apply.
Every qualified original witness remains a conflict veto. Pose, gyro, reference
and bridge histories are retained. A bridge cannot become a retained anchor.
The association rule is explicitly different; it is not described as preserving
the original descriptor gates. The controller retains the original floor
registration, mapping, mission, residual state and learned action selection.

Require exact complete original decisions at frames 0–213 after normalizing
only the new controller name and enable flag. This includes every forecast,
score, veto, selected action, pose, map/mission receipt and causal residual.
Compare every prior request with the completed actual command tape. At 214,
require the fallback to retain exact original camera, continuity and reference
failure evidence. Revalidate both current dual-camera and floor-registered pose
contracts when the controller recovers. Require a current full prediction and
selection with matching requested action. A retained terminal failure is a
valid negative scientific result, not authorization to weaken limits or retry.
Unexpected earlier differences are terminal comparator failures, and the exact
candidate row is saved before reporting the failure.

Inputs and source identities are checked before and after execution. The new
local import graph is discovered narrowly, stopping at inherited bound sources;
this does not reclassify the inherited manifest as a newly proved closure.
The assigned model digest must match before and after, with no gradients.
Public input arrays must remain unchanged. Native state is not a controller
input; there is no simulator, training, checkpoint modification or hardware
motion in this replay. All existing failures and sealed custody remain intact.

One CPU replay, one OpenCV/BLAS/Torch thread, 8 GiB admission allowance,
256 MiB output allowance and the existing navigation storage reserve. These are
admission/storage checks, not an enforced process RAM limit. Refresh CPU, RAM,
GPU/VRAM, storage and competing jobs before launch; one separately owned native
reactive scene may run concurrently. Save hardware again at completion.
Output is exclusive `go2_direct_flow_maze01_prefix_v1_attempt_001` under the
existing navigation development artifact root. No automatic retries.

Run `scripts/replay_go2_direct_flow_maze01_prefix_v1.py` with the recorded
Genesis environment and deterministic single-thread environment. A successful
prefix is still insufficient to claim native tracking recovery or navigation.
Fresh physical execution with full prefix and raw audit remains necessary.
