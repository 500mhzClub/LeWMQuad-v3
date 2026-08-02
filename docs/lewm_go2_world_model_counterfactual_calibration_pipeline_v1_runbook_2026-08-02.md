# Go2 counterfactual calibration pipeline V1 runbook

Date: 2026-08-02

Status: source/runbook only. This document grants no execution, retry, resume,
refill, training, scientific-claim, navigation, promotion, or deployment
authority.

## Scope

The next runtime tranche remains exactly eight calibration-only scenes, two
states per scene, nine candidate actions plus one repeated-action sentinel per
state: 16 states and 160 total branches. Sentinel `group_index i` duplicates
requested primitive `i mod 9`; the 16 repeats therefore cover every primitive
at least once. The repeats are an exact deterministic-replay gate, not an
empirical noise estimate: the receipt checker already requires the duplicate
trajectory, endpoint, executed tape, and stored RGB receipt to be exact. The
prior HOLD/`forward_medium`-only sizing allocation is
superseded for this fresh, not-yet-authorized calibration.

The collector records the requested primitive and the future executed command
tape separately. Candidate model input is the requested action ID. The future
executed tape is outcome/audit evidence and must never be supplied to the
candidate predictor.

## Required metadata inputs

Prepare two ordinary, non-protected, caller-bound JSON files:

1. A scene panel with schema
   `lewm_go2_world_model_counterfactual_calibration_scene_panel_v1`. Its
   `scenes` list follows the eight canonical family names in source order. Each
scene binds exact `manifest.json` and `genesis_scene.json` files and has two
state declarations containing `state_id`, two `history_action_ids`, and a
world-frame `target_xy_m`. The builder opens each bound ordinary manifest,
requires the panel `scene_id` and `family` to agree, and requires both state
targets to equal the center of the lexicographically first landmark. Arbitrary
finite target coordinates are rejected.
2. A runtime contract with schema
   `lewm_go2_world_model_counterfactual_runtime_contract_v1`, containing the
   exact `runtime_bindings` and `execution_contract` accepted by the collector.

Neither document is an authority. Do not place either input in or below a
sealed/protected custody path.

## Build the exact plan

Fill the six pinned identity values, then run:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/build_go2_world_model_counterfactual_calibration_plan_v1.py \
  --attempt-id lewm-go2-wm-counterfactual-calibration-v1 \
  --output-root /home/andrewknowles/Workspace/LeWMQuad-v3/.generated/dev/lewm-go2-wm-counterfactual-calibration-v1 \
  --scene-panel <scene-panel.json> \
  --expected-scene-panel-sha256 <64-lowercase-hex> \
  --expected-scene-panel-byte-count <positive-integer> \
  --runtime-contract <runtime-contract.json> \
  --expected-runtime-contract-sha256 <64-lowercase-hex> \
  --expected-runtime-contract-byte-count <positive-integer> \
  --plan-output <calibration-plan.json>
```

The builder fails unless the plan has exactly 8 scenes, 16 states, 144
candidate branches, 16 sentinel branches, 48 context frames, and 160 targets.

## Authority and source-freeze gate

Execution remains blocked until all of the following exist at exact reviewed
paths:

- a commit containing the collector, checker, calibration analyzer, joiner,
  consumer, plan builder, focused tests, plan, and supervisor source;
- an independent source review binding that passes that commit and exact source
  closure;
- a `lewm_go2_world_model_counterfactual_calibration_execution_authority_v1`
  document with status `AUTHORIZED_ONE_EXACT_160_BRANCH_CALIBRATION`;
- resolved platform gates and an exact positive wall-time cap; and
- a fresh output root.

The authority must bind the plan, runtime, source commit, review, one-shot
attempt, 160-branch caps, and external supervisor. Reservation consumes the
single attempt; retry, resume, overwrite, and refill remain false.

The metadata-only authority helper fixes a 57-name source-role order covering
the runtime closure, plan/authority builders, supervisor, analyzer, joiner,
checker, contract, consumer, and focused tests. It verifies every current
source byte against the reviewed source commit. First emit a deliberately
non-passing review template:

```bash
python3 scripts/build_go2_world_model_counterfactual_calibration_authority_v1.py review-template \
  --source-commit <40-lowercase-hex-source-commit> \
  --output <calibration-source-review.json>
```

The template has status `PENDING_INDEPENDENT_REVIEW`, a nonempty blocking
finding, and explicit `REVIEWER_MUST_REPLACE` fields. The helper cannot emit a
passing review. An independent reviewer must inspect the committed closure,
replace those fields, set `PASS_SOURCE_ONLY_NOT_AUTHORITY`, and clear findings.
Commit the exact plan and independently passing review, bind their bytes and
SHA-256 values, then materialize authority only from an explicit authorizer:

```bash
python3 scripts/build_go2_world_model_counterfactual_calibration_authority_v1.py authority \
  --plan <calibration-plan.json> \
  --expected-plan-sha256 <64-lowercase-hex> \
  --expected-plan-byte-count <positive-integer> \
  --review <calibration-source-review.json> \
  --expected-review-sha256 <64-lowercase-hex> \
  --expected-review-byte-count <positive-integer> \
  --authorizer-identity <explicit-authorizer> \
  --authorizer-basis <basis> \
  --issued-at <ISO-8601> \
  --terminal-reviewer <reviewer> \
  --wall-seconds <positive-number> \
  --platform-basis <resolved-platform-gate-basis> \
  --output <calibration-authority.json>
```

Commit the exact authority document before execution. The collector and
supervisor both reject an authority, plan, review, or runtime source that does
not match its committed binding.

## Collector invocation after authority exists

Do not invoke the collector directly. The calibration supervisor removes
ambient accelerator/render selectors, installs the exact plan-bound map,
revalidates the bound Vulkan and EGL devices under the authority wall clock,
then runs the collector, receipt checker, and analyzer once. Invoke it exactly:

```bash
/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/run_go2_world_model_counterfactual_calibration_authorized_v1.py \
  --authority <calibration-authority.json> \
  --expected-authority-byte-count <positive-integer> \
  --expected-authority-sha256 <64-lowercase-hex>
```

The nonce is generated privately by the supervisor. A pre-reservation
preflight failure consumes no attempt. Once `reservation.json` is written, any
failure consumes the sole attempt and no retry or resume path exists.

The supervisor samples `mem_info_vram_used` for the uniquely matched
0x1002:0x7551 DRM device from immediately before collection through terminal
analysis. The terminal reports baseline bytes, peak bytes, peak delta, device
total, sample count, read errors, and the exact sysfs paths. This is explicitly
selected-device global usage and is not process-attributed; concurrent GPU
work can inflate it. Any sampler read error makes the attempt fail closed.

## Receipt check and calibration analysis

On success, the supervisor writes `physics_result.json`, `receipt_check.json`,
`calibration_receipt.json`, and `terminal_supervision.json` below the exact
attempt root. It prints the terminal binding and calibration decision. The
checker and analyzer are already complete at that point. The following manual
commands are audit/reproduction entry points, not additional execution:

```bash
python3 scripts/check_go2_world_model_counterfactual_pilot_v1.py \
  --manifest <physics_result.json> \
  --expected-file-sha256 <64-lowercase-hex> \
  --expected-byte-count <positive-integer> \
  --output <receipt-check.json>
```

If and only if that passes, derive tolerances and the terminal decision without
opening RGB leaves:

```bash
python3 scripts/analyze_go2_world_model_counterfactual_calibration_v1.py \
  --collection <physics_result.json> \
  --expected-collection-sha256 <64-lowercase-hex> \
  --expected-collection-byte-count <positive-integer> \
  --output <calibration-receipt.json>
```

The analyzer emits either `FREEZE_PILOT_CONTRACT` or
`STOP_SOURCE_REDESIGN`. Because the checker already requires duplicate
branches to be exact, progress/path tolerances are the fixed 1e-6 m numerical
resolution floor. The repeat panel verifies exact deterministic replay across
all nine requested primitives; it does not estimate stochastic variability.
The decision also requires at least two physical rank classes in every state.
The calibration receipt additionally aggregates actual context/target/total
PNG byte counts, external collection and per-stage wall times, complete
all-nine-action group yield, clipping/fall/tip counts, and checker-guaranteed
zero camera-invalid/incomplete counts. GPU memory remains independently bound
to the external terminal because the analyzer runs before that terminal is
written.

This reviewed source is intentionally calibration-only. It does not accept a
`bounded_wm_a_pilot` authority: a later pilot requires its own reviewed
supervisor and authority contract after the calibration decision is frozen.

## Visual-evidence boundary

The current pipeline validates camera quality and binds sequential endpoint-pose
replay to the physical endpoint. It does not establish parity with the textured
training RGB domain. Every calibration receipt therefore keeps
`visual_domain_fidelity_claimed=false` and
`eligible_for_visual_domain_parity_claim=false`. This limitation must remain
visible in any downstream evaluation.
