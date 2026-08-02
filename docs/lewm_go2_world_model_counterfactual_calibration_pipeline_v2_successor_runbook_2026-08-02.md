# Counterfactual calibration V2 successor runbook

Date: 2026-08-02

Status: source and exact plan prepared; no execution authority minted; no launch
authorized by this document.

## 1. Why this is a successor, not a retry

Calibration V1 is permanently consumed.  Its terminal-failure audit is:

- `docs/lewm_go2_world_model_counterfactual_calibration_v1_terminal_failure_result_2026-08-02.json`
- SHA-256 `f97d48d11f88819526b673de820ec1b69910b2d500b0fe5bfeccf7a99ea7d490`
- byte count `5126`

That audit binds the consumed terminal supervision receipt
`c5509f97c1d1cca27b7f283187ce7bf644579c4caa03eb1ccfcfda9c18e58315`
and the already-written failed physics result
`34ba69825322e34ebec0ccbab5f1a21fdd4ac60f99cc4fe5f70b158a7aaaaaa3`.
It grants no authority and explicitly forbids retry, resume, refill, overwrite,
and V1-root or V1-artifact reuse.

V2 therefore has a new identity and a fresh root:

- attempt ID: `lewm-go2-wm-counterfactual-calibration-v2`
- output root:
  `/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/dev/lewm-go2-wm-counterfactual-calibration-v2`
- exact plan:
  `docs/lewm_go2_world_model_counterfactual_calibration_exact_plan_v2_2026-08-02.json`
- plan SHA-256:
  `41ecf46d4b1a3a0a6e835af746a10af52cc8557ab8edc5b3c0080538ab27d6cc`
- plan byte count: `28596`

The V2 root must be absent at every pre-authority review and immediately before
supervised execution.  Nothing under the V1 attempt root may be copied,
linked, resumed, refilled, or treated as successful V2 output.  The ordinary,
pre-existing scene inputs named by the scene panel are immutable inputs rather
than V1 attempt artifacts and remain the selected calibration panel.

## 2. Frozen experiment identity

The V2 plan preserves the V1 scientific design exactly:

- eight ordinary development scenes, one per registered family;
- two action-history states per scene;
- all nine requested candidate actions per state;
- one deterministic sentinel duplicate per state;
- 144 candidate branches plus 16 sentinels, 160 total;
- 48 context frames plus 160 target frames, 208 stored RGB frames;
- the same lockstep prefix, branch horizon, action catalog, runtime bindings,
  scene bindings, candidate/sentinel allocation, target coordinates, render
  contract, and execution contract.

Candidate model input remains the requested candidate action only.  The future
executed command tape is an outcome receipt and must never become predictor
input.  Sentinel branches remain calibration controls and are not candidate
training examples.

## 3. Minimal source repair

The only scientific collection change is the disposition of repository-defined
low-information observations.  The exact allowed registry is:

- `low_rgb_texture`
- `near_wall_depth`
- `near_forward_geometry`

These names must equal
`lewm_genesis.lewm_genesis.vision_quality.LOW_INFO_REASON_NAMES` at runtime.
A frame whose complete reason set is within that registry is technically
retained.  It receives explicit `low_information` and `low_info_reasons` tags
in frame receipts, joined RGB artifacts, and the calibration receipt.  The
analyzer records overall, context, and target frame counts plus exact per-reason
counts, allowing downstream stratification without silently discarding the
near-wall/low-texture navigation regime.

Every reason outside that exact registry remains a hard failure.  This includes
technical RGB/depth corruption, malformed or non-finite arrays, and unresolved
camera-safety conditions.  An unknown or hard reason cannot be relabelled as
low information.  A mixed low-information plus hard-reason frame hard-fails.

The V2 supervisor also loads and binds an already-written `FAILED`
`physics_result.json` before writing terminal supervision.  This closes the V1
receipt-lineage defect without turning any failure into a retryable attempt.

## 4. Source review and authority boundary

No current file grants V2 execution authority.  First commit the exact reviewed
source closure intentionally.  Then generate a non-passing review template from
that frozen source commit:

```sh
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/build_go2_world_model_counterfactual_calibration_authority_v1.py \
  review-template \
  --source-commit SOURCE_COMMIT \
  --output docs/lewm_go2_world_model_counterfactual_calibration_source_review_v2_2026-08-02.json
```

The template is deliberately `PENDING_INDEPENDENT_REVIEW`.  A genuinely
independent reviewer must inspect the exact source bindings, tests, predecessor
failure audit, low-information policy, hard-failure policy, attempt freshness,
and future-tape isolation.  The helper cannot self-assert a passing review.

Only after a passing, exact, independently completed review exists may an
explicit authorizer mint the one-shot V2 authority:

```sh
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/build_go2_world_model_counterfactual_calibration_authority_v1.py \
  authority \
  --plan docs/lewm_go2_world_model_counterfactual_calibration_exact_plan_v2_2026-08-02.json \
  --expected-plan-sha256 41ecf46d4b1a3a0a6e835af746a10af52cc8557ab8edc5b3c0080538ab27d6cc \
  --expected-plan-byte-count 28596 \
  --review docs/lewm_go2_world_model_counterfactual_calibration_source_review_v2_2026-08-02.json \
  --expected-review-sha256 REVIEW_SHA256 \
  --expected-review-byte-count REVIEW_BYTE_COUNT \
  --predecessor-failure docs/lewm_go2_world_model_counterfactual_calibration_v1_terminal_failure_result_2026-08-02.json \
  --expected-predecessor-failure-sha256 f97d48d11f88819526b673de820ec1b69910b2d500b0fe5bfeccf7a99ea7d490 \
  --expected-predecessor-failure-byte-count 5126 \
  --authorizer-identity AUTHORIZER_IDENTITY \
  --authorizer-basis AUTHORIZER_BASIS \
  --issued-at ISO8601_TIMESTAMP \
  --terminal-reviewer TERMINAL_REVIEWER_IDENTITY \
  --wall-seconds EXPLICIT_POSITIVE_WALL_CAP \
  --platform-basis EXPLICIT_PLATFORM_GATE_BASIS \
  --output docs/lewm_go2_world_model_counterfactual_calibration_execution_authority_v2_2026-08-02.json
```

The resulting authority must have schema
`lewm_go2_world_model_counterfactual_calibration_execution_authority_v2` and
status `AUTHORIZED_ONE_EXACT_160_BRANCH_CALIBRATION_V2_SUCCESSOR`.  It binds the
V1 terminal-failure result both directly and through the reviewed source
closure.  The collector rejects a missing/mutated predecessor binding, a V1
attempt ID, or the V1 output root.

## 5. Pre-launch verification

Run the focused source suite:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .generated/venvs/genesis_render_vulkan/bin/python -m pytest -q \
  lewm/tests/test_go2_world_model_counterfactual_pilot_v1.py \
  lewm/tests/test_check_go2_world_model_counterfactual_pilot_v1.py \
  lewm/tests/test_analyze_go2_world_model_counterfactual_calibration_v1.py \
  lewm/tests/test_go2_world_model_counterfactual_consumers_v1.py \
  lewm/tests/test_join_go2_world_model_counterfactual_pilot_v1.py \
  lewm/tests/test_build_go2_world_model_counterfactual_calibration_plan_v1.py \
  lewm/tests/test_build_go2_world_model_counterfactual_calibration_authority_v1.py \
  lewm/tests/test_run_go2_world_model_counterfactual_calibration_authorized_v1.py
```

Before launching, verify all of the following:

1. The source commit and all review bindings equal the live files exactly.
2. The V2 output root is absent and contains no symlink prefix.
3. The plan, review, predecessor result, and authority match caller-supplied
   SHA-256 and byte-count bindings.
4. The authority contains `maximum_attempts: 1`, consumes the attempt at
   reservation, and forbids retry, resume, overwrite, and refill.
5. The exact 144-candidate/16-sentinel design and requested-action-only model
   input remain unchanged.
6. The selected EGL/Vulkan device and global VRAM counters pass the bound
   platform preflight.
7. No calibration process is already running and the one explicit wall-clock
   cap is adequate for collection, checking, and analysis.

## 6. Authorized launch shape

After all preceding gates pass and only with the exact minted authority, launch
the external supervisor once:

```sh
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/run_go2_world_model_counterfactual_calibration_authorized_v1.py \
  --authority docs/lewm_go2_world_model_counterfactual_calibration_execution_authority_v2_2026-08-02.json \
  --expected-authority-sha256 AUTHORITY_SHA256 \
  --expected-authority-byte-count AUTHORITY_BYTE_COUNT
```

Do not invoke the collector directly.  Do not repeat this command if a
reservation was written.  Any consumed failure requires a separately reviewed
successor; it does not reopen V2.

## 7. Terminal review

A successful supervisor run is still only
`COMPLETE_PENDING_TERMINAL_REVIEW`.  Terminal review must bind the V2 terminal
receipt, physics receipt, receipt-check report, calibration receipt, source
commit, predecessor failure, GPU-memory measurement, wall-clock measurement,
and exact attempt identity.  It must separately confirm that low-information
counts are visible and internally consistent and that hard-invalid frames are
zero.

Only a terminally reviewed calibration receipt whose decision is
`FREEZE_PILOT_CONTRACT` can satisfy the downstream bounded-experiment
calibration gate.  This runbook, a source review, and an execution authority do
not themselves establish that result or authorize model training, scientific
promotion, retry, resume, or production use.
