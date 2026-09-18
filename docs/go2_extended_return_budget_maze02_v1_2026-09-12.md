# Prospective 8,000-step chained single-pass native development trial V1

## Question and fixed scope

Test whether a fresh execution of the corrected direct/no-RGB learned
controller completes the outbound-and-return mission on reused development
maze 02 when its shared navigation budget increases from 4,000 to 8,000
steps. Preserve the completed 4,000-step negative: it reached the outbound
goal, made six return crossings, and exhausted its budget about 2.62 m from
home. A larger budget is an experiment, not an assumed explanation or success.

The launcher is `scripts/run_go2_extended_return_budget_maze02_v1.py`.
The exclusive output is `go2_extended_return_budget_maze02_v1_attempt_001`
under `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1`.
The case is `no_rgb_direct_extended_return_budget_maze_02`, layout 2,
with the original variant, condition and model assignment. This protocol and
the recursive source/test union must be hashed into the actual launch.
No existing output is overwritten, resumed or automatically retried.

## Required completed evidence

The original chained native result is fixed at SHA-256
`163e92d630d92f9f8633cd3823c7cbf282a92c81bde5d3db998e3a5a8c8b1849`.
The chained/single-pass complete recorded-history comparison must finish,
followed by the prospective controller-prefix comparison specified in
`docs/go2_extended_return_budget_controller_prefix_v1_2026-09-12.md`.
Supply the latter's actual completed result SHA with
`--controller-prefix-result-sha256`. There is no placeholder result, queued
execution or automatic waiter.

Require a successful budget-only prefix: all 4,004 public observations and
normalized decisions through the intervention are accounted for, every
earlier decision matches, and frame 4003 changes the old budget terminal to
a nonterminal candidate decision. An earlier difference, incomplete replay,
changed source/model, live predecessor owner or terminal predecessor failure
cannot admit native execution.

Initial admission authenticates the prefix launch, complete six-artifact
roster and result; the complete original native artifacts, result and raw
audit; and the exact source ancestry. It reuses the completed prefix's bound
public-packet and closed-output audit rather than reconstructing it again.
The completed positive result and unchanged full artifact/source hashes are
required. Later worker and parent input checks rehash these bound inputs
without rerunning older controllers, physics or predecessor physical-prefix
analyses. The fresh native run's physical prefix is reconstructed separately
in both worker and parent. All predecessor owners must have ended.

Development startup revision: the first launcher invocation was deliberately
interrupted during redundant input reconstruction, before any output or
simulation worker existed. The completed comparison and original native
evidence are preserved. This revision removes that duplicate reconstruction;
the controller, model, physical experiment and outcome audit are unchanged.

## Executed controller and physical configuration

Use a fresh `ExtendedReturnBudgetChainedController`, with fresh memory,
geometry and the unchanged assigned model SHA-256
`56799c99e2bb1fd5e5a591009b931c5b2e04741d3a3744e2346ce9896a2160dd`.
The single-pass controller optimization is adopted only after the required
complete equivalence evidence. The separately prepared single-read camera
acquisition change is not adopted in this trial.

Retain the original scene, renderer, sensing, gait, action set, measured-plane
estimator, floor and temporal gates, bridge allowances, predictive selection
and return objective. The maximum populations are 8,000 navigation steps,
8,013 command ticks, 8,014 observations and 401,400 physics samples. The
physical loop retains 750 settling samples and 50 samples per command at
2 ms per physics sample. Outbound arrival does not reset the shared budget.
Existing terminal, stop, drain and audit behavior is preserved.

Use one spawned CPU worker, one task per worker, one OpenCV and BLAS thread,
deterministic Torch, disabled OpenCL and the original deterministic environment.
No concurrent native run or named full CPU replay is admitted. The original
model remains in evaluation mode, unchanged and without gradients; there is
no training. Evaluator-only native pose remains outside controller inputs.
Physics remains paused during sensing and planning.

## Resource envelope and evidence persistence

Initial admission requires at least 64 GiB available RAM and 84 GiB free
artifact storage. Collection and audit guards sample around acquisition,
controller calls and raw sensor auditing. They retain a 16 GiB available-RAM
floor and a 48 GiB worker-RSS ceiling. The collection allowance is 28 GiB,
persistence allowance 8 GiB and audit allowance 8 GiB, retaining 40 GiB of
artifact space. Phase guards reserve downstream persistence and audit space;
their complete ordered sample populations and receipts are reconstructed.

A separate whole-worker ledger covers initial input checks, persisted
collection, collection artifact verification, audit writing, physical-prefix
accounting, worker validation, input reauthentication, final artifact hashes
and terminal-record writing. It enforces cumulative disk use of at most
44 GiB and checks both current RSS and the operating system's reported
process peak RSS against 48 GiB. A peak detected after terminal writing still
fails the worker RPC and prevents parent acceptance. Resource breaches latch;
recovered resources do not authorize continued controller execution.

These are sampled application checks, not operating-system memory limits,
guarantees of available memory between samples or guarantees that cleanup
can finish after exhaustion. Whole-worker accounting excludes parent result
serialization. After its independent prefix reconstruction and final hash
checks, the parent requires 16 GiB available RAM and 40 GiB plus 256 MiB
free disk. Each metadata JSON write is exclusive, finite and limited to
256 MiB. Parent and worker resource evidence is retained with the result.

## Complete raw audit and actual intervention

Retain the full collection and execute the complete raw sensor, command,
model-command replay, renderer witness, strict physical visibility and native
outcome audits using the larger bounded populations. Collecting an episode
or writing an audit is not a round-trip success. Success requires the original
joint native round-trip candidate and strict visibility criteria, with no
hard measurement failure. Preserve all negative outcomes.

When the actual intervention interval exists, authenticate all consumed
original/current raw artifacts and prospective-prefix evidence before and
after comparison. Reconstruct the first 4,004 public packets and complete
decisions against the prospective proof, including every pre-intervention
command and all 11 raw physics fields over 200,900 common physics samples.
Require the actual 50-sample boundary command interval to match each arm's
declared request. Do not infer or require equal physical outcomes after the
changed command, and do not feed subsequent observations to a retrospective
controller. The parent independently reconstructs this comparison after the
worker has ended and its artifacts are closed.

If execution stops before that interval is available, retain the complete
raw audit as an early negative. Record the unavailable observation, command
or physical interval explicitly; do not claim a reconstructed paired prefix
or success. If counts say the interval exists, an actual comparison failure
is terminal and cannot be replaced by an unavailable-prefix receipt.

## Terminal accounting and interpretation

The worker closes and hashes collection, resource, audit, readout and prefix
evidence, reauthenticates inputs and artifacts, and writes its terminal record.
The parent requires equality between that saved record and the returned
record, the complete reconstructed whole-worker ledger, the independently
verified raw outcome/prefix, and final input and output hashes.

`EXTENDED_RETURN_BUDGET_MAZE02_V1_COMPLETE` means execution and verification
completed; its measured success count can be zero. An operational or evidence
failure produces `TERMINAL_EXTENDED_RETURN_NATIVE_FAILURE`, preserving the
exclusive attempt and existing artifacts without automatic retry.

This is one reused development layout, zero new independent-layout trials
and one fixed model. It does not establish unseen-maze reliability, JEPA
benefit, predictive-planning or memory advantage, real-time operation,
hardware qualification or completion of the broader goal. A verified round
trip would support the next independent matched-layout experiments; a
negative result requires diagnosis from this actual execution. Sealed
benchmark custody remains unchanged.
