# First-interval feasibility prefix: preparation and submission

The actual-observation prefix runner and comparator are implemented, and27
focused tests passed in2.10s. Execution was submitted in session97118,
PID2474093. At the initial live inspection it was verifying input/source
bindings (29.29CPU seconds,1,350,176,768bytesRSS), before launch output. Poll
that same session; do not infer replay completion or retry from this note.

Protocol: `docs/go2_residual_first_interval_prefix_v1_2026-09-09.md`.
Input is the completed three-layout cohort, SHA-256
`a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720`;
the policy consumes only the fixed maze2 public packets and original command
history. Maximum504 observations, stopping at the first changed command,
changed terminal, or original terminal. Original forecasts, scoring and veto
records must remain exact. Complete observed/mission/residual state must match;
only the declared controller metadata, alternative selection receipt and
action/wait accounting may differ. A newly eligible hold may change counters
without changing the actual command, and its first selected-action difference
is reported separately. No post-divergence outcome is inferred.

New source SHA-256:

- `lewm/residual_first_interval_prefix_development.py`:
  `d5139900b2ea3afaaa613f370ddf8966e0a9fd25bf2673baf6dacd5861ea6e61`.
- `scripts/replay_go2_residual_first_interval_prefix_v1.py`:
  `bdb152fa2d571251bcb2f81640c15ddacf793e687b2e37a5e10b478c7936ec79`.
- `lewm/tests/test_residual_first_interval_prefix_development.py`:
  `c3defd2c36038ca2d7c0cf7466c0310bc10ebcc918fed61507ed2f14505f86e8`.
- Protocol:
  `0b882056d3d78649ecb2a66087114c938fbfc6087d2e7ecd5711a42184977de7`.

Source inventory preparation returned1666 paths: the1658 inherited baseline
bindings plus8 explicitly discovered component, prefix, test and protocol
paths. This is not a whole-tree export or a newly claimed recursive closure
of all inherited files. The earlier29 feasibility tests remain separate;
the27 new tests exercise actual replay boundary control with synthetic packet
providers and fault injection, plus complete-state comparison and negative
cohort admission. They cover command and terminal boundaries, original terminal,
frame limit, mutated public input, changed weights, selected hold accounting,
stale pending forecast, and changes to raw predictions, utility, surface/path
vetoes, observed map, mission, residuals, metadata and commands. These tests
do not substitute for the submitted actual model/RGBD replay.

Hardware refreshed before submission:16physical/32logical CPUs, all32 in
affinity,3.4%CPU busy,80.369GBavailableRAM,90.931GBartifactfree,
21.360GBworkspacefree. Integrated GPU2.147GBtotal/0.413GBused and discrete
GPU34.209GBtotal/1.399GBused, both0%busy. Reactive parent2472753 and
worker2473157 were the substantive competitors; workerRSS2.630GB. One
sequential CPU replay is admitted alongside that separately owned scene/audit,
with one OpenCV/BLAS thread,8GiB memory capacity admission and256MiB output
allowance plus40GiB artifact reserve. The runner refreshes resources after
lengthy input validation before creating its exclusive output. Capacity
admissions are not enforced OS limits.

Command submitted with bytecode disabled, hash seed0 and explicit local
package paths, through the existing Genesis environment:

```sh
.generated/venvs/genesis_rocm_0_4_6_v1/bin/python scripts/replay_go2_residual_first_interval_prefix_v1.py
```

Keep the source definition fixed once submitted. A successful prefix would
still need a separate fresh native attempt and complete raw physical audit.
The reactive and planning-memory comparison queue remains unchanged.
