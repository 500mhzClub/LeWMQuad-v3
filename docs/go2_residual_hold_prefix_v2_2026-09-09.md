# Hold reconsideration V2: larger output envelope, identical replay

V1 stopped on its output ceiling after recording 1,185 unchanged decisions,
frames 0–1184. Its compressed stream is 134,243,265 bytes, exceeding the original
128 MiB stream ceiling within a 256 MiB total allowance. This was an output
accounting failure, not a policy mismatch or a completed no-intervention result.
Preserve the V1 launch, all recorded decisions, failure and frozen sources.

Require the exact preserved V1 artifacts:

- launch.json: `b344a274f4649943c24d0e6350c5d8cf00a2820a5cb6283ad9ba3e6dd222a725`.
- context_decisions.jsonl.gz: `b7d386890457e4c29e91660ddf2b19c72f4ac151daf3c7713673786bd753b33d`.
- failure.json: `548bad1eef69df9c2afd51accd3df9d2db17ef7aaf90939c3a45b990cd55e085`.

Recheck V1 with its original input verifier and require all 1,185 saved decisions
to remain unchanged. Authenticate the same completed residual native attempt,
all inherited artifacts/sources/environment and the same assigned JEPA model.
Merge the new source closure over the frozen V1 closure; reject conflicts.

V2 starts a fresh model and controller at observation zero. Its replay function
is AST-identical to V1. Keep the original first-changed-command/terminal stop,
maximum 3,004 observations, exact full decision comparison, every geometry,
surface and phase gate, strict utility improvement, model and input checks.
Never resume V1 state, consume later observations after a change, alter the
policy or infer unexecuted outcomes. A complete unchanged replay remains a
valid negative result; V1's truncated replay does not establish that result.

Only the output envelope increases: 2 GiB total admission and a 1 GiB compressed
stream ceiling. The complete original native stream is 347,142,791 bytes; this
provides roughly three times that stream size and separate persistence headroom.
The unchanged memory admission is 8 GiB and the unchanged reserve is 40 GiB.
Refresh full hardware/resources before submission and output creation. One CPU
replay with single numerical threads may overlap independent jobs only with
measured memory headroom. No native scene, model training, real-robot action,
deletion, source export, sealed access, threshold change or recovery of old state.

Exclusive output: `go2_residual_hold_prefix_v2_attempt_001`. Run the separately
named V2 launcher with the same completed native-result SHA. Both V1 and V2
artifacts remain independent. Retain any new failure; navigation and real-time
qualification remain false regardless of replay outcome.
