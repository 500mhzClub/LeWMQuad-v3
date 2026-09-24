# Direct-flow maze1 prefix V2: live validation correction

Carry forward the full scientific definition and resource/scope limits in
`docs/go2_direct_flow_maze01_prefix_v1_2026-09-09.md`. V1 terminated at its extra
runner validation after serializing a live identity tuple to a JSON list.
Preserve that attempt, its exact sources and all three failure artifacts bound
in `docs/go2_direct_flow_maze01_prefix_v1_failure_2026-09-09.md`.

V2 changes only the runner validation order: retain the live controller return,
serialize a separate recorded value, require that value to equal the exact
serialization of the live decision, and run live raw/registered pose contracts
on the live return. Do not relax identity typing or change observer, camera
association, controller, model, correction, reference/bridge limits, mapper,
mission, comparator or input observations. There is no forecast/command change
relative to the V1 candidate by design; the complete rerun is required because
V1 did not reach final input or weight authentication. The fixed original
comparison remains through frame213, with final failure/recovery at214 and no
observation215 consumed. A retained scientific failure is a valid negative.

Fresh model/controller from episode start, exactly one sequential CPU replay,
same deterministic single-thread environment, same 8 GiB RAM admission and
256 MiB output allowances plus existing storage reserve. Refresh full hardware
and competing workloads before launch. One separately owned native reactive
scene may remain active. All inputs, predecessor artifacts and frozen source
bindings are verified before and after. No native execution, model training,
hardware motion, sealed access, existing-attempt edits or automatic retries.

Runner `scripts/replay_go2_direct_flow_maze01_prefix_v2.py`; exclusive output
`go2_direct_flow_maze01_prefix_v2_attempt_001`. Run focused live/serialized
validation tests before submission. Preserve the entire original V1 result as
a terminal validation failure; a V2 success cannot retroactively complete V1.
