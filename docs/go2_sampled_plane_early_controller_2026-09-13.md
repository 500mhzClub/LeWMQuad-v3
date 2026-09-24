# Sampled plane extraction: early controller result

The new plane candidate extractor evaluates only the 76,800 image mesh quads
surrounding the original 19,200 stride-four candidate pixels, rather than all
306,081 image quads. Every candidate still requires its original four quads,
nine valid pixels, normal/planarity tests and reference/body height tests.
Full depth input validation remains. Near floating-point gate boundaries the
helper falls back to the original dense extractor. It is not a replacement
for the full floor-coverage index used by mapping.

Twenty focused tests passed in 2.08 seconds. They compare exact candidate
points and masks for both camera mounts, missing pixels, noisy depth, a depth
step, random depth, empty frames, strided arrays and alignment boundaries.
An invalid pixel outside the sampled population is still rejected.

A fresh development composition uses the helper in visual tracking and floor
registration. On frames 0–12 of the completed stop-conditioned maze02 recording,
both the original and candidate controllers exactly matched every complete
recorded decision after removing the candidate's explicit implementation flag.
This used the same saved no-RGB direct predictor; the controller still consumes
RGB-D. The model state remained unchanged. There was no simulator or command
execution in this comparison.

After the first three observations, ten alternating-order paired calls gave:

| Measurement | Original | Sampled extraction |
| --- | ---: | ---: |
| Median controller call | 476.397 ms | 415.388 ms |
| Total over ten calls | 4.771 s | 4.304 s |

Total controller time decreased 9.806%. The candidate was faster on nine of
the ten timed frames; on frame 12 it took 583 ms versus 470 ms. This is a small
early-history measurement on a shared host with the original supervised audit
running concurrently. It is not a stable whole-history speed estimate. Packet
acquisition and native physics are outside the timed controller calls.

The comparison completed in session 37373 with exit code 0. Per-frame times,
source identities, model identity and the input launch identity are retained
in `go2_sampled_plane_early_controller_2026-09-13.json`. The diagnostic source
is `scripts/compare_sampled_plane_early_controller_development.py`; the helper
and composition are `lewm/sampled_plane_candidates_development.py` and
`lewm/sampled_plane_stop_conditioned_controller_development.py`.

This candidate is not installed in the live or queued navigation runners.
The measured median still exceeds the 100 ms target by more than four times.
Full-history equivalence, continuous execution, realistic sensing and native
navigation with this implementation remain unproven. The result supports
reducing unnecessary floor extraction work, but that optimization alone cannot
resolve the controller's timing deficit.
