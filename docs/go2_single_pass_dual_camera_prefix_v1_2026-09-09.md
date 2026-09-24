# Exact map optimization on the completed dual-camera prefix

The dual-camera native episode67399 remains frozen and unchanged. Prepare a
separate empty-state controller that replaces only its eight measured-bound
map indices with the existing packed-owned/single-pass implementation. Preserve
the dual-camera observer, floor registration, measured settling mission,
learned model, residual, selector, memory semantics, thresholds and all decision
labels. No observation or command is added to the live simulation.

Use the completed dual-camera controller prefix result
62037c72cc85546a5379299226aabf3d61f4715d92ce64dfb503a0ebd98909c1,
and its exact closed tenth input collection/full native audit. Bind the prior
single-pass full-controller equality result
4687d67fbb53fce3b29a122e379b51805fa685b4c39e3e817e3a5862374d342e
as previous implementation evidence, not proof for the new integration.

One new model/controller processes exactly the1873 admitted observations,
starting empty at frame0. Compare every complete decision exactly against the
completed dual-camera prefix, including the auxiliary intervention at1872 and
its requested turn[0,0,-0.45]. Require all earlier commands to match the actual
tenth executed tape. Do not consume any later recorded observation under the
changed final command. Require unchanged input arrays and final model/source/
artifact identities. No metadata normalization or scientific-field waiver.

Use scripts/replay_go2_single_pass_dual_camera_prefix_v1.py with the actual
--prefix-result-sha256, --preflight-only before execution, exclusive root
go2_single_pass_dual_camera_prefix_v1_attempt_001. Inspect CPU affinity/activity,
GPU/VRAM, RAM/storage and competition. One CPU/numerical/OpenCV thread alongside
the one existing native scene; no second scene. Admit8GiB replay RAM plus32GiB
native headroom and1GiB replay output plus10GiB native collection/1GiB persistence
above40GiB reserve. These are planning checks, not OS limits. Enforce the1GiB
compressed replay-stream allowance and preserve any failure without retry.

Record processing time as operational telemetry only: this is not a paired
speed comparison, a native-loop deadline result, or proof of incremental gains
over packed-only maps. The previous timing comparison still missed every100ms
deadline. Do not adopt this candidate in the live native scene. Any future
adoption requires completed equality evidence and separate prospective scope.
No new navigation/round-trip, independent-layout, matched-contribution or
hardware qualification follows from this replay.
