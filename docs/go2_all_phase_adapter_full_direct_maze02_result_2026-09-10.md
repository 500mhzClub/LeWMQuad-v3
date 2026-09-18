# Completed direct adapter maze-2 episode: no traversal

The full direct-model adapter completed collection and its original full raw
sensor/model/command audit. Independent authentication reconstructed its startup
and complete native readout, checked 1,908 source bindings and 18,124 artifact
bindings, and verified the parent completion and unchanged assigned model.
The development ledger now contains 40 completed audited episodes and zero
verified round trips. This is case three of the six-case batch.

Across 3,000 navigation decisions, it selected 2,760 holds (92%), 113 left arcs,
122 right turns and five left turns. There were no arrivals, crossed maze edges,
return traversal or verified round trip. The mission exhausted its tick budget
and completed ten terminal zero commands. Collection contains 3,014 observations,
3,013 command intervals and 151,400 physics samples, with zero native contacts.

Raw reconstruction and replay pass, but strict physical visibility fails at
primary-camera frame 1320. No auxiliary failure is reported. The separate
hard-measurement failed-frame list is empty; that does not override the strict
visibility failure. The failed sampled comparison reports maximum depth error
0.5422852084036338 m. Preserve this failure; the episode is not qualified sensing
or successful navigation evidence.

Acquisition plus controller processing across all 3,014 observations has median
2580.7136165 ms, p95 3911.94352005 ms and maximum 5313.996806 ms. Every observation
exceeds the 100 ms command interval. Median iteration time including receipt
persistence is 2632.084715 ms. Physics was paused during computation; these
measurements do not establish real-time or hardware readiness.

Independent authentication did not rerun the full raw audit, training-input
admission, neural inference or native collection. It authenticated the original
completed audit and reconstructed startup and physics-based evaluation from
its recorded inputs. The original worker has ended and the parent advanced to
`all_phase_no_rgb_jepa_residual_maze_02`. The three no-RGB cases and subsequent
frontier, hold-reorientation and contact-score experiments remain outstanding.

Artifact root: `go2_all_phase_adapter_maze02_matched_native_v1_attempt_001`.
Case: `all_phase_full_direct_residual_maze_02`.

- Worker terminal: `e757508fb28a0983890f62857d66d4105a862069e823c3bea13488dc32ad341d`.
- Raw audit: `ef559bbf890e14850962c1a63b7ee5878cfa51b4a2f6ec666baa72528a00ca50`.
- Readout: `b25ada81a3516d115b783acc78bb867bb8f01b26d98623f599270cd17b91f02c`.
- Parent completion: `6294647bb03411fb7478a6e0eb1df4ff1291feb13be900e44014d08416170de9`.
- Model: `8cf81bec6c67261df9d25b56d2f6546abd887735ab8977021bb54a3c1963f1df`.
- Independent verification: `35c30a45137912829e114306bb8ff63e471d6d48d8b2e4c7e2879cd717219a30`,
  `go2_all_phase_adapter_full_direct_maze02_verification_2026-09-10.json`.

The goal remains incomplete: this result establishes no reliable navigation,
JEPA/planning/memory advantage, or deployment readiness.
