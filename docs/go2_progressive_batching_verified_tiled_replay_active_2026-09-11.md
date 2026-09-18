# Progressive batching verified; tiled geometry replay active

The progressive batching replay and its completion verifier both finished
successfully. Every original decision, 1,425 model forecasts and seven retained
state checks matched over 1,428 observations. Original strict sensing-failure
evidence at frame 1173 remains; no physics or navigation outcome was added.

Paired total controller time decreased from 920.825586 s to 912.811313 s,
only **0.8703355741565555%**. The median decreased from 615.188 ms to 600.824 ms.
All 1,425 navigation observations still exceeded 100 ms. The early,
repeated-hold and late ten-frame windows had greater candidate total time;
the late median increased from 773.042 ms to 868.392 ms. Preserve these mixed
shared-host timings. Progressive batching has not been adopted into navigation.

- Result SHA-256:
  `bc8f387a3b93aa946b77aba7756bb62f25e51e1b48d7ac2cef5720d251628888`.
- Completion document:
  `go2_progressive_batched_floor_controller_completion_verification_2026-09-11.json`,
  SHA-256 `f40606e05fb6e4dd52188875ae98739631ef1a34e22ac3e8e32256bd7099e2b7`.
- Replay session 69904 and automatic completion-watch session 30020 exited zero.
  Owners 2856701 / creation 1789134105.22 and 2860273 / creation 1789135792.48
  have ended. Do not launch the progressive completion checker again.

The next paired replay isolates tiled dense floor geometry, using progressive
controllers in both arms. Its recorded component probe showed byte-exact floor
arrays and 20.56–57.45% auxiliary-camera total-time reductions, with substantial
shared-host timing variation. That does not yet establish a controller gain.

- Runner: `scripts/replay_go2_tiled_density_progressive_floor_late_history_v1.py`.
- Artifact root: `go2_tiled_density_progressive_floor_late_history_v1_attempt_001`.
- Owner: PID 2862056, creation 1789136599.37, session 99411; original boot
  `1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`. Command is the original generated
  Python interpreter, `-B`, and that runner path.
- Launch SHA-256:
  `8ad6b00ce598b26b2058c9c904a51a908f2f63f4398b6eb5b46e8484b6ca45dc`;
  2,337 source bindings. Initial full input admission and paired frame 0 passed.
- Execution document:
  `go2_tiled_density_progressive_floor_controller_replay_execution_2026-09-11.json`,
  SHA-256 `0d449415fdd73ce8160e73a243ec285612dcf4dc1d3be37e240480f95a7bb762`.
- Launcher preparation:
  `go2_tiled_density_progressive_floor_launcher_preparation_2026-09-11.json`,
  SHA-256 `bd164f5f7152ff74a2230bbceab3aec565f4399faf74a6ced5e8e73d18018af6`.
  Admission tests: 22 passed in 2.60 s, session 72096. Harness tests: 16 passed
  in 4.96 s, session 79354. Source preflight: session 90836, exit zero.
- Completion checker:
  `scripts/verify_go2_tiled_density_progressive_floor_controller_completion_v1.py`.
  Its execution hash is now bound. Tests: 18 passed in 2.41 s, session 65337.
  Preparation SHA-256:
  `a2bbac6f70acf89bd32ab7b2c5775476693783974ca32bf697932c770497b8dc`;
  2,340 sources, including an actual rejection of the still-live owner.
- Automatic verification is recorded in
  `go2_tiled_density_progressive_floor_completion_watch_execution_2026-09-11.json`.
  Inspect its exact owner before attempting any manual completion check.

Next, monitor the exact tiled replay and its completion watcher. Preserve any
failure and require the complete verification before interpreting its timing
result. Keep one full CPU replay at a time. No placeholder bindings remain in
the launched tiled runner or its prepared checker.

The extended-budget native launcher remains active in repeated nested input
verification, with advancing CPU and read counters and no native child at the
last check. Keep the original queue: extended budget, sustained turn,
contact/flow, chained anchors. No verified round trip, independent-maze
comparison, real-time qualification or hardware result has been added.
