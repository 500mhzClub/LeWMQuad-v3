# New layout family: valid measurements, failed short-horizon task-design gate

The fresh96episode collection and every raw audit completed. Both roles pass the
prospective RGB/body measurement gate. The **frozen action-design gate fails**:
the four-second progress threshold accepts arcs that have not yet traversed the
panel, and a constant left arc succeeds in all eight transfer layout/appearance
contexts. This score cannot establish that scene-dependent selection is needed
on the transfer task. No fitting or learned-policy execution occurred. The
overall navigation goal remains active.

The completed separate causal readout retains768 planned windows, including684
actual contexts and84 missing contexts. It preserves the failed design gate and
reports `ready_for_separately_frozen_learning=false`. The policy-stream checker
that requires that flag was **not launched**; its output root does not exist.
Its implementation and synthetic tests are source-only work.

[Compact verified readout](go2_geometry_progress_family_scientific_readout_2026-09-08.json)
contains complete role counts, measurement diagnostics, timing and identities.

## What the new native evidence shows

Four parameter clusters, each with mirrored left/right openings, two matched
appearances and six fixed actions. Training and geometry-transfer each have two
disjoint parameter clusters, four layouts and48 episodes. Mirrored siblings and
windows within an episode are dependent. These are local obstruction tasks;
zero independent complete-maze evaluations occurred.

| Parameter cluster | Role | Mirrored arc reversal, both appearances | Contact-free progressing actions |
| --- | --- | --- | --- |
| 00, panel x630mm | training | Pass | Matching arc in each opening |
| 01, panel x690mm | training | Fail | Left opening: left arc; right opening: both arcs |
| 02, panel x660mm | geometry transfer | Fail | Left opening: left arc; right opening: both arcs |
| 03, panel x720mm | geometry transfer | Fail | Both arcs in both openings |

Training has10 successful local-progress candidate episodes and14 contact stops;
transfer has14 successful candidates and10 contact stops. All hold/pure-turn
controls fail the150mm distance-reduction threshold. Forward contacts the panel
in all16 contexts. There is no constant successful action across all training
contexts, but **left arc succeeds in8/8 transfer contexts**. Thus the data still
contains some geometry-dependent contact outcomes; the complete preregistered
design criterion and the proposed transfer discrimination are not established.

The successful left/right arcs reduce goal distance by313.25/351.08mm. Their
native endpoints remain before passing the panels; even the closest endpoint is
about848.92mm from the1.2m goal. The threshold tests short-term progress, not
passage traversal or arrival. Moving panels farther from the start lets a wrong
arc stop before its eventual obstruction becomes relevant. The observed gait
also yields different left/right displacements, so a geometrically mirrored
layout does not guarantee symmetric finite-horizon contact outcomes.

[Native base-centre paths](/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_geometry_progress_family_causal_v1_attempt_001/native_paths.png)
([SVG](/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_geometry_progress_family_causal_v1_attempt_001/native_paths.svg)).
All eight layouts and both appearances are plotted. Crosses mark contact-stop
endpoints; curves show base centres, not swept robot volumes. The figure was
visually inspected. It makes the gap between local progress and the downstream
goal explicit.

## Acquisition, measurement and causal accounting

All96 worker episodes completed collection and raw audit:72 complete horizons,
24 measured contact stops,3,740 camera/decision packets and254,726 native2ms
samples. All18,035 terminal-bound artifacts are present, totaling5,724,575,574
bytes including worker/audit products. No episode was dropped or retried.

Every frame passes the declared stable-interior/near-occlusion contract. Minimum
stable-ray coverage is2,790; maximum stable residual is0.34067mm. There are zero
stable bad frames, clipped opaque rays or falsely public-valid near-surface rays.
Two strict boundary failures remain explicit: frame27 of
`family_episode_008` and `family_episode_057`, both cluster00/right-open/left-turn,
one per appearance. They contribute two boundary bad rays. The prospective
footprint gate did not classify those boundary pixels as certified metric
interiors. Strict depth failure is preserved; no depth-navigation qualification
is granted.

The original initial-departure targets retain768 slots:660 valid native motion
targets,660 actual future-image packets,768 known contact labels and108 positive
contacts. All96 initial tensors exactly match the existing generalized adapter.
There are16 distinct initial RGB histories, one body history and one control
history. All96 prefix comparisons, including16 references and80 nonreferences,
are exactly equal within layout/appearance strata. Independent-episode provenance
remains explicit.

The separately frozen causal derivation uses offsets0,5,...,35 ticks after each
original departure, with the last four actual100ms policy packets and the known
remaining suffix of the original command plan. Unknown suffixes are masked;
they are not assumed braking commands. Native state is target-only.

| Causal accounting | Training | Geometry transfer |
| --- | ---: | ---: |
| Planned windows |384|384|
| Actual contexts |336|348|
| Actual initial contexts |48|48|
| Actual moving contexts |288|300|
| Missing actual contexts |48|36|
| Recorded known horizon slots |1,608|1,642|
| Valid motion / actual future-image targets |1,352|1,446|
| Known positive contacts |256|196|

All452 positive contact targets retain missing future motion and images. Missing
contexts and inactive plan suffixes remain explicit. No post-stop future or
sibling context was substituted. Moving suffixes broaden the recorded context
distribution; they do not establish repeated predictive replanning or grant a
passing design/learning-ready flag.

## Throughput, tests and source state

The prior separately frozen eight-episode scaling comparison showed identical
array/pixel signatures and3.3423x speedup with four fresh processes. Its records
remain excluded from this dataset. See
[the completed scaling report](go2_geometry_progress_family_scaling_result_2026-09-08.md).
All96 dataset episodes were fresh and used that exact benchmarked source.

The collection/audit phase took1,106.07s, about18.43min, with four workers and one
episode per process.65 recorded resource observations showed at most15.8% total
CPU activity, about7.98GB combined Python RSS, at least77.25GB available RAM and
97.09GB free artifact storage. The12GiB planned output allowance and40GiB reserve
were respected. This scheduling improvement did not make sensing real time:
median observation/control time was138.66ms, and3,522/3,740 decisions exceeded
100ms before adding physics-tick cost. Physics paused during processing.

45 distinct new focused tests passed: execution/concurrency8, generalized
initial tensors9, cohort accounting6, causal windows13, balanced learning view4,
and policy stream5. The existing five family-geometry tests passed in the prior
turn. Source-equivalence tests retain the reviewed physical constructor,
collection loop, exact command audit and per-frame checks while introducing
explicit new family/root arguments. No frozen predecessor was monkeypatched.

Completed execution sources: `scripts/geometry_progress_family_runtime_development.py`,
`scripts/geometry_progress_family_episode_development.py`,
`scripts/geometry_progress_family_session_development.py`,
`scripts/geometry_progress_family_audit_development.py`,
`scripts/run_go2_geometry_progress_family_v1.py`,
`lewm/geometry_progress_family_accounting_development.py`, and
`lewm/geometry_progress_family_learning_sample_development.py`.

Completed separate derivation: `lewm/geometry_progress_family_causal_windows_development.py`,
`scripts/read_go2_geometry_progress_family_causal_v1.py`, and
`scripts/geometry_progress_family_plot_development.py`.

Source-only fitting preparation: `lewm/geometry_progress_family_learning_view_development.py`
and `scripts/geometry_progress_family_policy_stream_development.py`. Tests verify
equal episode weighting despite different numbers of available windows, no
transfer-role training draws, past-only inference, bound cache admission and
private cached samples. The whole-population checker
`scripts/check_go2_geometry_progress_family_policy_stream_v1.py` and its protocol
exist, but were not launched because their explicit design prerequisite is false.
No real-data fitting schedule, optimizer output or checkpoint was produced.

## Terminal identities

All roots are under
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1`.
All launched processes are terminal; no retry or resumption.

| Root | File | SHA-256 |
| --- | --- | --- |
| `go2_geometry_progress_family_v1_attempt_001` | `launch.json` | `bd851bc0ef6f078152b86b1998e2c0baa61ce18c2a584e6900ee41df86ec0216` |
| same | `collection_terminal.json` | `039001d8172ca2c5efb547e1813ad506f549bc0b946d1b3fea047989c8acc949` |
| same | `result.json` | `376b2eeacfd5e741ba6b7a0e1b5e04399f782d16943d66ffdff92eada93ef0fb` |
| `go2_geometry_progress_family_causal_v1_attempt_001` | `launch.json` | `a02fae4600391aa6f85ffc85da45a207ed31fe6532e63f5c3011460c8a27cf16` |
| same | `result.json` | `37bd88d43fd0ebdcee80282d38695f82a41b83313d08158d076f5ad4ce7145a7` |
| same | `windows.json` | `e73c1c06b0afd8f58e96c7e6ee53da43368c1b3dde0fa4954e0385acdcf8f632` |
| same | `tensor_index.json` | `181d05960261ccc52364bbdfdaac711ef47705a0de4b6021b874693c23a860f6` |
| same | `native_paths.png` | `e58b14db34eeb4bced6f078bd33260185790cc5e32e4bb96f267e50e623d7978` |
| same | `native_paths.svg` | `355fbde5253fb3389f21739c96af2d2db9049ebb3a35d238b996430f98c2d5a3` |

Collection/scaling bind833 source paths; the causal derivation binds838.
Later learning-view/stream/checker preparation is outside those execution
bindings. All new work remains local/uncommitted.

## Ordered continuation toward the actual goal

1. Preserve both measurement passes and the failed short-horizon design gate.
   The current policy-stream checker correctly requires a missing design pass;
   do not launch it unchanged, relabel the result, select a favorable subset,
   or repeat this96episode cohort.
2. Make the next decision-task experiment measure actual passage traversal and
   downstream goal-reaching over multiple actions under a fixed mission budget.
   The learned forecast may remain four seconds; the navigation task need not
   finish within one forecast. Freeze new criteria before fresh physical episodes.
3. Keep measurement eligibility distinct from the claim that this local success
   score establishes a need for RGB/planning. Any later use of these valid
   motion/contact records needs an explicit new experiment and honest retention
   of the failed original claim; this result grants no passing learning-ready flag.
4. Reuse the implemented causal suffix, exact-command and policy-only interfaces.
   The eventual fitting study must compare matched RGB/no-RGB and
   direct/rollout/JEPA arms, then use an authenticated fitted model for real online
   command selection. Avoid another unchanged short-prefix success study.
5. Integrate the existing joint observer and continuous local executor, then test
   memory, physical returns and reactive/nonpredictive baselines on independent
   mazes. Account for moving-context distribution gaps and actual timing.
6. Leave the long-term goal active. Learned online goal-reaching, contribution
   ablations, independent complete-maze reliability, realistic sensing/timing and
   bounded hardware evidence remain unproven.
