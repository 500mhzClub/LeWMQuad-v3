# Capability qualification and video plan, 28 September 2026

Fixed before any validation run or outcome. It applies the [handoff](go2_navigation_capability_handoff_2026-09-28.md) §6 and the pre-registration to the frozen harness `v4_completed_support` (`82b7b604…`). It takes effect only if that harness passes its C0 gate.

## Runs

- **Episodes.** C1–C4 run on validation episodes 10/0–29/0 (20 each). C0 runs on 10/0–19/0. These are one fresh attempt each, under the fixed reduced design. There are no retries and no substitutions.
- **Owner.** `scripts/run_go2_capability_completed_support_v4_validation_development.py`. It uses the unchanged run owner, arrival and clearance readers, and hash retention. C0 is bound to the passing 00/0 replay receipt and qualified at closeout under the [prefix erratum](go2_navigation_capability_oracle_prefix_erratum_2026-09-28.md).
- **Concurrency.** Up to five concurrent owners, with at most two C3 at a time, admitted by free memory. This is output-preserving under handoff §5.7. Video replays re-verify determinism for the selected episodes.
- **Stops.** A C0 validity problem, a technical or closeout failure, or any disallowed contact or hard violation stops new launches. Controller failures such as pose loss or timeout are results.

## Analysis

- **Unit.** The maze, with one episode per maze.
- **Metrics per controller** (handoff §5.4):
  - beacon, home and round-trip success;
  - SPL per leg;
  - time to beacon and home (median and IQR over successes);
  - disallowed contacts, hard and operating-margin violations, and the FK interval bound;
  - stall rate by phase;
  - decision latency (median and p95);
  - wall time per episode;
  - the failure taxonomy.
- **Intervals.** 95% intervals come from the pre-registered paired maze bootstrap: resample maze IDs, 10,000 replicates, seed 2026092519, percentile intervals. Raw counts are always shown.
- **Capability criterion.** A controller is capable if round-trip success is at least 16/20 (80%) with zero disallowed contacts, judged on the point estimate. C0 is diagnostic, on 10 episodes.
- **Label.** Capability qualification, not paper results. The render-provenance caveat applies to C3 and C4.

## Videos

- **Controllers.** C1–C4. C0 is optional and not produced.
- **Episode selection.** Use the lowest-ID validation episode that all of C1–C4 completed as round trips. If there is none, use each controller's lowest-ID success. A controller with no success gets its lowest-ID episode, labelled a failure.
- **Replay verification**, required before publishing. Re-simulate from the episode seed and recorded commands on the frozen V4 session. Then:
  - every consumed RGB and depth packet record (pixel and packet SHA-256, depth noise, arrays, timestamps) must match bitwise;
  - the V4 controller runtime must recompute every decision. C1 and C2 use their actual non-neural model; C3 and C4 use the recorded prediction-slot outputs, with identical candidate tapes asserted;
  - selected actions, every dispatch command and reason, and every applied command must be identical;
  - the native trace must be exact (publication tolerance 1 mm / 0.1°).

  Any failure means the video is not published, and the reason is reported.
- **Chase camera.** Rendered in a separate commands-only replay session with a fixed offset (−0.6 m behind, 3.0 m above, smoothing 0.8), as in the pipeline-test chase revision.
- **Frames.** 1920×1080 at 30 fps in simulated time, H.264, yuv420p.
- **Panels:**
  - egocentric: the consumed 10-Hz RGB, held between frames;
  - chase;
  - true-maze minimap with home, beacon, pose, heading and a leg-coloured trajectory;
  - HUD with controller, harness, maze and episode, simulated time, phase, current action, stall indicator, arrivals and validation success rate.
- **Extras:**
  - a 4× cut, labelled in-frame, for missions over 120 s;
  - a 2×2 composite if the four controllers share an episode;
  - a 12-keyframe contact sheet per video;
  - metadata JSON binding the episode, harness, models, replay verification and video hashes.
- **Storage.** All intermediates and outputs go under `videos/` on RecoveryStorage.
