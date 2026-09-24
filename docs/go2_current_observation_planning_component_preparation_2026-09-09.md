# Planning-map persistence ablation component preparation

Implemented a separately named CurrentObservationPlanningController with an
explicit current paired-observation view for the unchanged selector chain.
The baseline controller and all running/frozen sources remain unchanged.
No recorded-packet replay, native collector or experiment is launched here.

The map extension captures actual current floor coverage from the existing
paired observation context and reconstructs current occupied cells with each
camera's original arithmetic order. It returns the original accumulated-map
receipt unchanged and retains every contact partition, patch and later-floor
ledger entry. Current cells must be subsets of the admitted accumulated map.
The selector receives this view for route, terminal target, scanning, nominal
action/path and reentry queries; contact queries use the original surface
memory. Each view rejects stale/failed access and exposes immutable cell maps.

The intervention is precisely accumulated planning-cell queries. Tracking,
floor anchor, contact history, learned four-frame input, executed residual,
mission/settling and selector scan state remain. Metadata explicitly records
accumulated_planning_cells_queried=False and selector_scan_state_retained=True.
This is not a fully memoryless controller, a runtime optimization or evidence
that planning memory improves navigation.

Validation:

- Initial tests10170closed12pass4.40s. Metadata was then narrowed to explicitly
  name accumulated cell queries and retained scan state; the full focused suite
  was repeated. Final tests74597closed12pass4.28s.
- Tests include changing an observed route into a current-view frontier,
  re-observed cells retaining current membership despite old first-witness
  frames, no fallback to historical floor on an empty current view, immutable
  maps, stale/failed/unwitnessed rejection and original obstacle height/grid
  rules. A synthetic paired-camera sequence with temporary floor loss verifies
  identical complete baseline decision fields outside explicit variant metadata,
  retained maps, routes, residuals and actual articulated contact-query results.
  The complete selector chain uses the current view while issuing all six
  contact queries to the original persistent surface object.
- Source preparation81669closed successfully:1658prepared paths, all1654live
  native sources checked and unchanged. No model checkpoint or scene loaded.

Final component source identities:

- lewm/current_observation_planning_map_development.py:
  c33d3c432341e8c4f1cae6d98a6c45262acd3917a6b5a8c6438e9bf44b92666d
- lewm/current_observation_planning_controller_development.py:
  200beb6dc69fd80937ce000e413e3c869851beb1a4c90a30c106a049e3223d21
- lewm/tests/test_current_observation_planning_development.py:
  a7e662ab5f21da497eaaf7816e7cdcedd89c82a0dbd96eb1a354ef203b37ab57
- docs/go2_current_observation_planning_component_v1_2026-09-09.md:
  3acec7285c13bbac654e998786dbd7725e7f5acae993b295add4bface40e5fa1

Next for this component: define and freeze its causal actual-observation prefix
comparison, verify unchanged state and executed commands until the first policy
difference, stop before the following old observation, then prepare fresh
native execution and full raw auditing. No existing queued learned/reactive
cohort is changed. Preserve the current experiment/readout execution order.

Current native19976worker2447506confirmedliveRl98.9%CPU at87m27s,
CPU86m32s,RSS10,649,388KiB. Collection is complete at mission budget;
raw audit/prefix remain pending. No root result/failure, case audit or worker
terminal exists. Goal active/unachieved;zero verified round trips.
