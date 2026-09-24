# Go2 two-resolution navigation development V4 foundation author handoff

Date: 2026-07-14

Author role: `/root/navigation_v4_foundation`

Status: **PARTIAL SOURCE-CLOSURE AUTHOR CANDIDATE ONLY; NO REVIEW PASS; NO
POST-G2 BINDING, CHECKPOINT, DATA, `.generated`, GPU, TRAINING, G2,
DEVELOPMENT-RUN, GENESIS, RUNTIME, HARDWARE, ROBUSTNESS, HELD-OUT, PROMOTION,
OR BENCHMARK AUTHORITY**

## Frozen governing contract

The author read the complete 627-line contract before implementation and
verified its exact SHA-256:

| Path | SHA-256 |
|---|---|
| `docs/lewm_go2_two_resolution_navigation_development_v4_successor_contract_2026-07-14.md` | `707a4996574f1251ddf4c26703e2aa24b4310b038bc3556c78c20cfef65ba646` |

This handoff implements only ordered source gates 1 through 3's foundation
slice: closed trace/hash schemas, the synthetic-only one-frame accounting and
lease runtime, and the two detached auxiliary-head architectures. It is not a
complete V4 controller.

## Exact frozen author candidate

### New source bytes

| Required source | SHA-256 | Frozen responsibility |
|---|---|---|
| `lewm/benchmarks/go2_navigation_development_trace_v1.py` | `2e16e83e874f24534bf06824d39f7bacb7d52636f6878b77e20b24256611b795` | closed `ControllerEpisodeBindingV1`, `ResetReceiptV1`, `NavigationTickRecordV1`, `ActualOpenLedgerV1`, and `ControllerTraceV1`; duplicate-aware canonical JSON; finite canonical binary64; exact content/tick/trace/ledger chains |
| `lewm/benchmarks/qualified_shared_v5_navigation_runtime_v1.py` | `faf053c06f6f9c3699c622aa9b48e50adf230d17c90277ffb17a4ed6133de9d9` | synthetic-only fake-frame runtime; exact reset/session/tick/revision bindings; one-frame/one-token-call accounting; one target batch; at-most-one G4 call; exact-object single-use leases; terminal replay/stale/mutation behavior |
| `lewm/models/shared_v5_target_observation_head_v1.py` | `b6ebfbaed22fc1203e5d77a8756423c8c0e8638d194c35694525488bb8a57d90` | detached cached patch/BEV architecture producing one canonical batched four-colour presence/bearing/range/uncertainty/quality output; zero encoder/preprocessor ownership |
| `lewm/models/two_resolution_frontier_value_head_v1.py` | `8023d7a90f7a1e2dec9894aa2b8a8d9ebceabddbdf2dd63ccf1d0c6066aef8eb` | learned finite scalar value per immutable exact candidate row; exact row-order preservation and first-row tie break; zero encoder/candidate-generator/fallback ownership |

### New focused synthetic/mock tests

| Test source | SHA-256 |
|---|---|
| `lewm/tests/test_go2_navigation_development_trace_v1.py` | `fc5ac4535fb4c389c47768bd50d46ed0a6deb0ba682e59e64df37f8c0d598cce` |
| `lewm/tests/test_qualified_shared_v5_navigation_runtime_v1.py` | `8b4326a5fc301ba6aafd0c9049b8c02a1f790da277273011311e0dce92b0a29b` |
| `lewm/tests/test_shared_v5_target_observation_head_v1.py` | `95150b37694394788735b412540439a9fc7b80022c98eca7020a34fed078b2a7` |
| `lewm/tests/test_two_resolution_frontier_value_head_v1.py` | `31f1ac4b3ab325e79f57753f129de05f007787ff12cbff3230cc3cca71b50e28` |
| `lewm/tests/test_go2_navigation_development_v4_source_closure.py` | `693b2759a90459d8cd32fce3b1c8841e7d981d92d6d4fc562e3b45ed43e16e98` |

These test files use only synthetic temporary/in-memory tensors and mock
identities. They are production-ineligible and grant no artifact identity.

## Author test execution

The exact focused set completed once after the final source changes:

```text
PYTHONPATH=/usr/lib/python3/dist-packages
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
CUDA_VISIBLE_DEVICES=
HIP_VISIBLE_DEVICES=
ROCR_VISIBLE_DEVICES=
HSA_VISIBLE_DEVICES=
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
/home/andrewknowles/TinyQuadJEPA/bin/python -m pytest -q -p no:cacheprovider \
  lewm/tests/test_go2_navigation_development_trace_v1.py \
  lewm/tests/test_qualified_shared_v5_navigation_runtime_v1.py \
  lewm/tests/test_shared_v5_target_observation_head_v1.py \
  lewm/tests/test_two_resolution_frontier_value_head_v1.py \
  lewm/tests/test_go2_navigation_development_v4_source_closure.py
```

Result: `28 passed in 0.65s`.

No worker pool was created; numerical runtimes were limited to one thread;
accelerator selectors were blank; external pytest plugins and pytest cache
were disabled. The first attempted invocation found no `pytest` in the
TinyQuadJEPA environment and executed no tests. The recorded successful
invocation added only the host's `/usr/lib/python3/dist-packages` to expose
pytest 7.4.4 while retaining the TinyQuadJEPA interpreter and Torch runtime.

## Closed source graph for this slice

The new source import graph is fixed as follows:

- `go2_navigation_development_trace_v1.py` imports Python standard-library
  modules only;
- `shared_v5_target_observation_head_v1.py` imports Python standard-library
  modules and Torch only;
- `two_resolution_frontier_value_head_v1.py` imports Python standard-library
  modules and Torch only; and
- `qualified_shared_v5_navigation_runtime_v1.py` imports the three exact new
  foundation modules it needs, the target and G4 output types, and Torch.

No new source imports Shared V5 itself, `VisionEncoder`, an older navigation
integration, a synthetic target issuer, native projection, Genesis, a scene,
manifest, observer, evaluator, scorer, plugin loader, or dynamic source
selector. No new source contains a filesystem/artifact open or checkpoint
load. The runtime constructor accepts only the exact
`FakeSharedV5FrameBackendV1` with an explicit synthetic-test capability, and
the fake backend itself requires an explicit synthetic-test capability.

The exact observed per-admitted-tick invariant in this slice is:

```text
observation_tick_count
  == shared_frame_outcome_count
  == shared_v5_forward_frame_call_count
  == vision_encoder_forward_tokens_call_count
  == target_four_color_batch_count
  == rgb_decode_call_count
  == rgb_preprocess_call_count

g4_value_head_call_count <= observation_tick_count
extra_rgb_decode_or_preprocess_count == 0
target_head_owned_encoder_count == 0
g4_head_owned_encoder_count == 0
```

Receipt/lease payloads do not expose a whole four-colour output to one colour
consumer. Each target lease contains only its exact canonical colour slice.
The physical-view foundation payload contains only revision/content bindings,
not cached visual features. The learned G4 lease is the sole auxiliary lease
that exposes the already-counted detached cache.

## Retained anchor rehash

The author rehashed the retained source anchors after the final focused test.
Every byte matches the governing contract:

| Retained source | SHA-256 |
|---|---|
| `lewm/models/shared_observable_camera_ray_jepa_v5.py` | `b438295d7ec5cb0897cc953a229f461da7fca16322c4c936555d37833a36e4b9` |
| `lewm/planning/revisioned_physical_configuration_memory.py` | `bb05f957e0443e0c1e8405042b97c61948746a66040e84690e12b0a10887d483` |
| `lewm/planning/two_resolution_configuration_projection_v2.py` | `3c858a89170f78a73f401c9534e231f24d6d91bb0469ea95eb00002158146107` |
| `lewm/planning/native_learned_physical_projection_v5.py` | `5ccd22e83c83a4c41db11286d31d417fe7af5615ebd7e62e51d7719d5378eca1` |
| `lewm/planning/two_resolution_frontier_viewpoint_v2.py` | `5c84e79e558f51b75b00cf2baa26d7860302d6e3912ac14432dfc010efdc4f82` |
| `lewm/planning/two_resolution_target_evidence_v1.py` | `f731b848f6b7ced3b07e11d4f9edca81daa8c66f083f9d503ed069809e38a9a2` |
| `lewm/planning/two_resolution_reversible_target_belief_v1.py` | `6d17d06718df355893fa7a6f2f1f735fcf835933178e53c554f4d60181ae96c3` |
| `lewm/planning/two_resolution_target_router_v2.py` | `c8e071d239d1b9894028752fdc090cc2e1be9273f6f9de5a7c7b4d147741b6d2` |
| `lewm/planning/two_resolution_world_waypoint_adapter_v2.py` | `9b710c6f6044bfefd3fd52bcdbb55a52f890b1fdc6c00629029bbf5a670e8fc1` |
| `lewm/planning/two_resolution_navigation_development_integration_v3.py` | `6d8b00aa8ffaa0117efc01baa218cadd299a871732e86d2751e51463520d6523` |
| `lewm/benchmarks/go2_physical_claim_evaluator.py` | `7ea003160ea03da6e989cb76124501b1e7de8571bf8586870b9c8dd7b42f04df` |
| `lewm/benchmarks/go2_physical_claim_trace.py` | `a41f1fa22f5a90503c82db459ccc9520af334173d416bac0b090308d69cc8fb3` |
| `lewm/benchmarks/go2_physical_claim_observer.py` | `1db940a49f01313b23c5d37699796b52da776a3a5c88bf3af1381d7d58103e30` |
| `lewm/benchmarks/strict_result_scorer.py` | `d4d4fb6ddff297faaf86e0e1ec9590a35deca2f0f2b0e92fe46dfc31fdd187c2` |

## Exact unresolved/null production bindings

Every post-G2/trained/runtime identity remains `None` in source and `null` in
this handoff:

```json
{
  "selected_post_g2_shared_v5_checkpoint_file_sha256": null,
  "selected_post_g2_shared_v5_model_state_sha256": null,
  "passed_g2_report_sha256": null,
  "g2_candidate_publication_sha256": null,
  "physical_calibration_sha256": null,
  "physical_admission_thresholds_sha256": null,
  "target_head_architecture_sha256": null,
  "target_head_config_sha256": null,
  "target_head_checkpoint_sha256": null,
  "target_head_calibration_sha256": null,
  "g4_head_architecture_sha256": null,
  "g4_head_config_sha256": null,
  "g4_head_checkpoint_sha256": null,
  "g4_head_calibration_sha256": null,
  "g4_candidate_baseline_configuration_sha256": null,
  "follower_configuration_sha256": null,
  "command_block_configuration_sha256": null,
  "qualified_runtime_sha256": null,
  "controller_binding_sha256": null,
  "captured_source_graph_sha256": null,
  "development_panel_sha256": null,
  "development_result_sha256": null,
  "robustness_result_sha256": null,
  "heldout_identity": null
}
```

The architecture source/config objects compute local content commitments for
synthetic tests, but none is promoted into a production binding. Plausible
hash-shaped mock values in tests are not artifact identities.

## Remaining V4 source closure

The following contract rows remain unimplemented and unresolved; none is
satisfied by this foundation:

- `lewm/planning/native_learned_physical_projection_v6.py`;
- `lewm/planning/two_resolution_frontier_viewpoint_v3.py`;
- `lewm/planning/two_resolution_target_evidence_v2.py`;
- `lewm/planning/two_resolution_reversible_target_belief_v2.py`;
- `lewm/planning/two_resolution_target_router_v3.py`;
- `lewm/planning/revision_bound_waypoint_follower_v1.py`;
- `lewm/planning/two_resolution_navigation_development_integration_v4.py`;
- `lewm/benchmarks/genesis_external_command_episode_v1.py`;
- `lewm/benchmarks/go2_visibility_opportunity_observer_v1.py`;
- `scripts/execute_go2_two_resolution_navigation_development_v4.py`;
- every corresponding focused test; and
- the complete mock end-to-end V4 test.

No real physical transaction, target evidence/posterior, G4 candidate issuer,
router, follower, coordinator, broker, observer, launcher, reset transport, or
development episode exists in this slice.

## Required next gate

A different agent must rehash these exact source/test/document bytes and the
retained anchors, inspect the closed graph and unresolved globals, and run its
own bounded CPU-only adversarial source review. This author has not performed
that review and makes no PASS claim. Even a different-agent PASS may say only
**source-ready for continuation of the separately authorized V4 source
closure**; it cannot authorize post-G2 binding, training, G2, development,
runtime, production, benchmarking, or held-out access.
