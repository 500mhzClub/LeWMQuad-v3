# Combined measured-plane perception and completed performance implementation

Run `scripts/replay_go2_measured_plane_single_pass_prefix_v1.py` only after the
exact existing deferred-memo replay and its automatic completion checker have
ended successfully. Their completion is scheduling evidence; the pending
copier optimization is not included in this candidate.

Compare `MeasuredPlaneResidualController` with the separate
`MeasuredPlaneSinglePassController`, which installs the same measured-plane
motion observer in the completed `SinglePassBodyProjectedController`. This
combines existing verified performance changes with the revised estimator.
The original queued native controller remains unchanged.

Use the exact original corrected no-RGB direct model, SHA-256
`56799c99e2bb1fd5e5a591009b931c5b2e04741d3a3744e2346ce9896a2160dd`,
loaded independently twice with disjoint tensor storage and no training.
Replay all 123 public observations from the completed measured-plane controller
comparison, bound by completion `ca03a74ce91eba199ae485acc6ee5729872557e428b42e8ef805a99135dc4bbf`.
Both new arms must reproduce every complete recorded measured-plane decision,
including all 120 forecasts. Normalize only existing performance flags/root
labels; retain all measured-plane and scientific evidence. Any mismatch fails
the attempt and is preserved.

Observation 122 precedes the first changed physical command: old hold versus
prospective right turn. Do not consume observation 123 or claim any following
physical outcome. All earlier commands must match the original executed tape.
Compare the complete retained memory, floor, occupied map, residual and public
history at frames 0, 3, 61 and 122, using the existing ten exact type-tag
normalizations and no additional field exclusions.

Alternate controller execution order by frame. Time only each `observe` call;
hashing, input reconstruction and state comparisons are outside the interval.
Report all 120 post-warmup timings, including deadline misses, without profiler
or outcome-based window selection. Concurrent queue activity means this is not
an isolated benchmark. The short prefix cannot establish late-history speed.

Require the original deterministic CPU environment, 64 GiB available RAM and
43 GiB artifact free space. Source preflight hashes source and completed
component evidence, checks resources, creates no output and runs no model.
Execution authenticates the original worker's entire artifact roster before
and after, checks model states, and reconstructs every compact output row
against the complete recorded decision stream. It does not rerun unrelated
training ancestry or alter the native queue.

Exclusive root: `go2_measured_plane_single_pass_prefix_v1_attempt_001` under the
existing navigation artifact volume. Preserve failures; no retry, overwrite,
resume, native execution, navigation qualification, real-time qualification,
hardware claim or goal-completion claim follows from this replay.
