# Fresh measured-plane maze-02 pilot V1

Run one fresh simulated episode using the originally assigned corrected no-RGB
direct world model and MeasuredPlaneResidualController. This follows the
verified 3,838-frame observer history and 123-frame paired trained-controller
replay. The latter first changes the requested command at frame 122, from hold
to right turn. A new physical episode is necessary to observe its consequences.

Keep the original maze-02 scene, public mission, physics, gait, renderer,
primary/auxiliary RGB-D packets, gyro history, floor and temporal gates,
planner, persistent memory and 4,000-tick navigation budget. The collection and
raw audit use private copies of the original extended-budget function globals,
substituting only the measured-plane controller constructor. Their function
code, original dependencies and inherited collection-status label are retained.
The launcher definition and complete controller decisions identify the new
estimator explicitly. No code in the original attempt is patched.

The model state is
`56799c99e2bb1fd5e5a591009b931c5b2e04741d3a3744e2346ce9896a2160dd`.
Admit the exact controller completion
`ca03a74ce91eba199ae485acc6ee5729872557e428b42e8ef805a99135dc4bbf`
and result `8385e643b776865a44d9271404e8c05a8acc46b37e7ff9bc4b8bf48396e93047`,
and the already admitted original extended-budget worker. All owners must be
ended before their completed evidence is used.

Start only after the existing chained-anchor native waiter completes and its
exact owner ends. That waiter follows extended-budget, sustained-turn and
contact/flow work. Require its actual completed result identity, frozen launch
`cf6703e83197d6c75df35b2b53834a47a53a293a06c64737ce94ade2ac0b87c1`,
complete raw-verification and physical-prefix receipt, and linked child result.
Accept a completed negative scientific result; never bypass or retry a failed
queue. Use these receipts for scheduling evidence. Do not rerun unrelated
training or treat the predecessor's outcome as evidence for this new episode.

Require 32 GiB available RAM and 55 GiB artifact-volume free space. Use one
fresh spawned CPU scene worker with one OpenCV/BLAS/Torch thread and the
original deterministic renderer environment. Record current CPU affinity,
utilization, GPU/VRAM, RAM and volume state before launch and monitor resources
every worker wait cycle. Capacity checks are not OS resource limits. Preserve
the original 14 GiB collection allowance and persistence reserve.

Record every command, observation, physical sample, renderer witness and full
controller decision. Reconstruct all raw sensors and replay the complete
controller/model episode; retain physical contact, visibility, timing and
mission evaluation. Check unchanged model states. Verify the first 123 fresh
decisions against the candidate replay, common public packets and the shared
6,850 physical samples through the observation before the first changed
command. Require the fresh episode to complete the actual right-turn command
at frame 122. Its following 50 samples may differ from the predecessor; do not
claim they match. Later outcomes come from the new episode and full audit.

If collection ends before the comparison boundary, retain its complete raw
audit as an early negative result without claiming prefix completion. Preserve
integrity failures and all partial evidence. The round-trip criterion remains
the conjunction of the original physical round-trip evaluation, strict
visibility and no hard measurement failures. A completed audit is not a
successful round trip.

Use the exclusive root `go2_measured_plane_maze02_pilot_v1_attempt_001` in the
existing navigation development artifact volume. Bind all recursive sources,
tests, protocol and admission evidence at launch and check them before and
after execution. No retry, resume, source revision, deletion or overwrite is
part of this attempt. This reused development layout does not establish unseen-
maze generalization, JEPA advantage, real-time operation or hardware readiness.
Physics remains paused during controller computation. The full goal remains
incomplete regardless of this pilot's result.
