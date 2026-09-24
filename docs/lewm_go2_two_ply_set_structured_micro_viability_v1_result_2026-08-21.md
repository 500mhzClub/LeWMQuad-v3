# TWO_PLY_SET_STRUCTURED_MICRO_VIABILITY_V1 result

Date: 2026-08-21
Source commit: `94693e5a1b102de52782cef642d87ea89965d67f`

## Outcome

`TWO_PLY_SUCCESSOR_EVIDENCE_RECONSTRUCTION_BLOCKER`

This is a pre-training evidence blocker, not `TRUE_SUCCESSOR_SET_VIABILITY_NO_SIGNAL` and not a model no-go. No model was initialized, trained, calibrated, or evaluated.

The existing current-state ledger preserves each current candidate's contact label and aggregate successor safe-action count. For non-legacy roots, however, it does not preserve the actual successor snapshot, successor planning-time sensor row, or fourteen individual successor-action contact rows. Deterministic reconstruction is not equivalent to the frozen evidence:

- development-training `viability-fit-1-01`, lateral-right: replay `5`, frozen `13` safe next actions;
- development-training `viability-fit-2-02`, lateral-right: replay `7`, frozen `5`;
- internal-calibration `viability-fit-2-04`, lateral-right: replay `14`, frozen `0`.

The internal-calibration discrepancy triggers the registered technical-consistency stop. Continuing would require either changing a frozen label or pairing it with a different successor observation.

## Frozen bindings and attribution

- The exact 128 training, 24 internal-calibration, and 24 development-held-out identities remain those in `.generated/development_micro_viability_model_screen_v1/development_internal_calibration_v1.json`, SHA-256 `d4148595ae1b3336eb7b5b597e78f83303c79af41bdc2e3210cd9c39b1c72db2`.
- The oracle row ledger's actual SHA-256 is `0a273a3f464f770ccf8d28a1c6c3d9ddad63efdb767c1a63175ddcb479a18eea`. The prompt supplied 65 hexadecimal characters (`...eea5`), so its final `5` cannot be part of a SHA-256 digest.
- The predecessor model ledger SHA-256 is `555ba6d2678e543cf78d6a53977eceeaa5bddf60a6c16c2510ee028db9f7cba2`.

The completed predecessor attribution is:

| Split | Contact AUC | Contact AP | Direct nonviability AUC | Direct nonviability AP |
|---|---:|---:|---:|---:|
| Training | 0.996272 | 0.983056 | 0.990741 | 0.755886 |
| Internal calibration | 0.738640 | 0.673844 | 0.435148 | 0.212752 |
| Development held-out | 0.485721 | 0.240629 | 0.492609 | 0.081523 |

Supported descriptive attributions are `FIT_TO_HELDOUT_GENERALISATION_FAILURE` and `DIRECT_NONVIABILITY_TARGET_MISALIGNMENT`; `MODEL_UNDERFIT` is not supported by the training ordering.

## Bounded materialisation

Materialisation stopped on the first evaluation-role aggregate mismatch. Before the stop it completed 95 of 176 state identities and 16,716 replayed branches:

| Role | Completed states | Current transitions | Compatible successor groups | Compatible successor transitions | Total compatible transitions |
|---|---:|---:|---:|---:|---:|
| Training | 77 | 1,078 | 928 | 12,992 | 14,070 |
| Internal calibration | 18 | 252 | 169 | 2,366 | 2,618 |
| Development held-out | 0 | 0 | 0 | 0 | 0 |

Two technically incompatible training successor groups were excluded rather than relabelled. Legacy frozen rows were used as authoritative labels; their reconstruction inventory contains 16 current-verdict and 168 next-verdict differences. This inventory is descriptive and does not overwrite any frozen outcome.

Completed-state runtime summed across workers was 4,499.302 s; diagnostic wall span was 902.029 s. Persisted storage was 23,155,559 bytes. The ignored result artifact is `.generated/two_ply_set_structured_micro_viability_v1/result.json`, file SHA-256 `3ec84a20b2a665a722ec17bc0c317e8c166d487c46db94eeb999a5d125975a07`.

## Gates and compute

Because the calibration successor evidence is not reconstructible under the frozen target:

- parameter count: not applicable;
- seed/checkpoint: none;
- temperature/threshold: none;
- contact and safe-action-count metrics: not evaluated;
- viable/nonviable state decisions and route outcomes: not evaluated;
- two-ply compute benchmark: not run.

`MICRO_VIABILITY_COMPUTE_SIGNAL` from the predecessor remains preserved. `REPLANNING_INTERFACE_UNRESOLVED` also remains preserved; no inference-only compute result qualifies observation delivery or command acknowledgement.

## Decision

Do not collect `FRESH_MICRO_VIABILITY_PANEL_V2` and do not train the set-structured contact evaluator from the current evidence.

The exact next experiment is `TWO_PLY_SUCCESSOR_TRANSITION_CORPUS_REPAIR_V1`. At branch creation it must persist:

1. the actual successor planning-time observation and full controller state;
2. all fourteen individual next-action physics-rate contact labels;
3. the aggregate safe-action count bound to those rows; and
4. byte-stable regeneration evidence before model initialization.

Only after that repair may the single-seed shared contact evaluator be trained. No successor-state predictor or direct nonviability classifier is justified before the contact evaluator receives coherent transition evidence.

## Claims and execution boundary

This result concerns a development simulated-contact evidence contract only. It establishes no learned viability, closed-loop control, material-impact safety, physical Go2 parity, or emergency-stop capability.

No fresh panel, model seed, JEPA predictor, successor predictor, utility model, memory, novelty, routing, beacon discovery, or navigation system was trained or executed. No process remained running at finalization.
