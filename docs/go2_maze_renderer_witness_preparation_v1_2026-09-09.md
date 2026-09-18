# Future maze renderer provenance recorder prepared

New scripts/maze_renderer_witness_development.py and
scripts/renderer_witness_dual_camera_maze_session_development.py record actual
context identity, sampling and precision at the original primary capture and
the paired capture endpoint after restoring the primary pose. They preserve
the inherited raw capture calls and exact returned public packet objects.
Renderer/native pose metadata remains evaluator-only. Query-side pose/clock/
sample changes and context drift latch a failure; partial evidence is saved
even if parent persistence fails. Cached acquisition reuse creates no new
witness. The witness audit binds all raw image hashes, sample/time identities
and the primary optical camera pose using the existing convention checker.

The post-pair endpoint follows the original auxiliary segmentation render and
camera-pose restoration. It does not assert that the framebuffer's contents
show the restored primary pose or reconstruct any per-draw shader arithmetic.
The installed renderer routes segmentation/depth-only rendering to its main
single-sample target, disables multisampling and leaves that draw target bound
after readback. Inspection34378 verified the actual renderer source against
the completed real-Camera integration probe; SHA256
1dc7c47b17a82c8aad7c2801bbfad7632dcaeeab8ecc8a382ebe06dee21a0b84.
This supports the selected query points but is not an actual maze-wrapper run
or proof of a rasterization error bound. Historical visibility failure909 and
uncertainty remain unchanged.

Tests90857 CLOSED13passed2.01s cover original capture/packet object preservation,
both endpoint records, no duplicate cached captures, query-induced physical/
camera mutations and exceptions, latched failures, context drift, inconsistent
raw hashes/poses/samples, incomplete witnesses, unproved precision claims and
persistence failure. These use synthetic context callbacks; the original
actual-Camera helper probe remains separate evidence.

The separately prepared three-observation integration consists of
scripts/renderer_witness_maze_probe_episode_development.py,
scripts/renderer_witness_maze_probe_comparison_development.py and
scripts/probe_go2_maze_renderer_witness_v1.py. Its collector differs from the
running dual-camera collector only by the new session/witness artifact, an
explicit three-observation boundary, required original zero commands and
scope/status metadata. It retains all900 physical samples across settling
and three warmup commands, with a declared acquisition cutoff rather than
inventing a navigation terminal. It reuses the unchanged full raw sensor/
model/command audit, then requires exact startup physics, complete decisions,
public packets and raw primary/auxiliary rasters against the completed current
native recording. A passing probe is not a new navigation episode.

Tests98697 CLOSED15passed2.29s cover completed-native admission, compatibility
with the original raw command audit for an explicit bounded probe, rejection
of relabeling it a terminated navigation episode, final-step inclusion and
old-future exclusion, and changed physics, packets, decisions, raw pixels or
commands. Collector source diff reviewed. CLI9968 CLOSED pass. Source39474
CLOSED pass:1631 prepared probe paths, compatible inherited camera evidence,
all1616 native and1624 optimization-replay frozen source bindings unchanged,
future probe root still absent. No actual probe preflight or scene was run.

Protocol:docs/go2_maze_renderer_witness_probe_v1_2026-09-09.md. After the current
native episode completes its full raw audit and prospective prefix comparison,
use its actual result hash for the prepared probe's --preflight-only. Inspect
current competition/resources before its single scene; do not overlap native
scenes. Native67399 and optimization2602 remain active and unmodified. The
known visibility failure and full-goal sensing, independent-layout, matched-
contribution, timing and hardware requirements remain unfinished.
