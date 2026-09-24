# Paired planning-memory outcome readout preparation

Prepared, not executed. The planning-memory native pilot and current baseline
readout must complete before this paired readout can run. It requires actual
--native-result-sha256 and --learned-readout-sha256 and binds both native
experiments, the completed baseline readout, their launches and merged sources.

Admission requires the same scene/mission/runtime/sensor/actuator/renderer and
budget settings, assigned model,1250sample/11packet native prefix, all eleven
prospective decisions and eight identical forecast banks. It checks the
declared retention of contact, localization, prediction/residual, mission/
settling and scan state. It does not label the intervention memoryless.

The candidate uses the original baseline actual-motion and outcome readout
calculations. Its trace additionally records current planning floor/occupied
cell counts. Paired metrics retain native arrival windows, outbound/return
traversals (including null), physical retracing, terminal quietness, physical
candidate pass, strict visibility, hard failures and verified success. Path
length, terminal coordinates/distances, duration, command counts, original
terminals and receipt-inclusive timings are preserved. No controller selection
or outcome rewriting occurs; no general memory advantage is claimed.

Validation:20focused tests passed in0.12s. They reject incomplete or mismatched
experiments/models/scenes/budgets/runtime, changed retained state and incomplete
physical/forecast prefixes. They preserve a physical candidate with failed
strict visibility as unverified and keep an absent return null. AST comparison
checks unchanged baseline motion/outcome calculations outside the explicitly
added planning trace and metadata fields. CLI54515closed successfully.

Source preparation18788closed successfully:1678prepared native sources,
1658prepared baseline readout sources,1688prepared paired readout sources;
all1670frozen planning-prefix sources (including1654live baseline sources)
unchanged. Both planning-native and planning-readout outputs remain absent.
No native preflight, scene, model load or readout execution.

New source SHA-256 identities:

- lewm/planning_memory_method_comparison_development.py:
  ee5be2be40f65ab8138efa4f28a35f5e3d5528805c0a6cdd0f580aa60e702d58
- scripts/read_go2_current_observation_planning_maze_pilot_v1.py:
  6df08933c2f9a21eed8248ff6eb0b4c1eae64ef0100298aeced861884f751ed4
- lewm/tests/test_planning_memory_method_comparison_development.py:
  21a36891fc9a11165943dc06e8b06e90d5343c528b239c91fca14f1ce32f207b
- docs/go2_current_observation_planning_maze_readout_v1_2026-09-09.md:
  93701f5c9ef1040a7eae175dcb3d89beed3dc04368e7a88cafed6d3bd98a1fa2

Existing order remains: current baseline audit/readout, independent learned
cohort, reactive maze0/paired readout, independent reactive cohort, planning-
memory native pilot and this paired readout. Refresh hardware assessment before
substantial jobs. The readout admits8GiBavailableRAM and128MiBabove40GiBreserve.

The original native19976worker2447506was confirmed activeRl99.1%CPU after
116m09s,CPU115m13s,RSS10,686,252KiB. No case audit, worker terminal, root
result or root failure existed. Continue the same full audit without restart.
Goal active/unachieved;zero verified round trips.
