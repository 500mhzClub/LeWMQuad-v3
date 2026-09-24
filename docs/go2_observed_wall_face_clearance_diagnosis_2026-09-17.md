# Why retained obstacle surfaces missed the reverse wall face

The exposed no-early-release failure now has a reconstructed map prefix. **The
map did not forget the wall or lose its cells. It retained the previously seen
face, while the nearer reverse face remained unobserved.** Clearance against
those stored points consequently overstated clearance to the physical wall
volume as the robot moved around its end.

This diagnosis concerns
`go2_view_arc_no_early_release_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.
The full failure, preceding exposed failure and original fifteen-run comparison
remain unchanged. No new native navigation was executed in this investigation.

## Reconstructed measured map

`scripts/replay_go2_no_early_release_map_entry_development.py` reconstructed the
160 recorded mapping updates through frame 636 from original delivered noisy
depth and saved registered poses. It used the same current-plane floor mapper,
fine obstacle cells and geometry parameters, with native geometry excluded.
Both camera noise digests were checked by the existing reader.

All 160 corresponding recorded map cell counts match, and every recorded
current-clearance value available in those plans matches within 1e-10 m.
No obstacle cell was removed. The original runtime did not persist complete
map hashes, so this is exact agreement on those saved witnesses, not a claim
that an unavailable original map-byte hash was checked. Tracking was not rerun.
Reconstruction took 16.95 seconds.

At planning frame 576, using map frame 572, the nearest stored cell was
`[199,-65]`, observed at frames 188,196,200,228,232,236,264,268,272,364,368,372,
376,384. Current stored clearance was 0.494188 m, while physical wall clearance
was 0.443980 m. The maximum pose error across the entire mission was 5.18 mm.

The post-execution wall-face comparison shows an 80-mm-thick wall. In the local
corner region, older observed cells occupy map y indices -64/-65, corresponding
to the previously visible face. They remain unchanged through map frame 596.
Reverse-face cells at indices -72/-73 appear later: by map 624 they extend only
to x index 187, by map 628 to 195, and by map 632 to 198 near the wall end.
The nearest stored cell therefore changes to `[195,-73]` at plan 632 and
`[198,-73]` at plan 636, when all six forecasts become blocked.

This agrees with the earlier raw-depth check: the nearest wall was behind both
cameras at sampled frames 560–600, although other scene pixels remained valid.
Its nearby reverse face entered the auxiliary view during the turn. The wall's
analytic dimensions and native pose were used only to interpret this completed
recording, not to construct the map or make controller decisions.

Evidence in the failure root:

- `map_entry_replay_v1.json`: update counts, plan comparisons, selected map
  snapshots and each fine cell's observation frames.
- `map_wall_face_readout_v1.json`: wall-face interpretation and bounded corner
  region counts.
- `map_wall_face_readout_v1.png` / `.svg`: visually inspected comparison of the
  retained face, later reverse-face observations and nominal footprint.

## Recorded footprint-coverage probe

The existing action guard checks predicted distance to observed obstacle hits;
it does not require the swept footprint to stay on observed floor. A saved-data
probe tested an additional signal: whether a candidate's 0.48-m swept disk
(existing 0.45-m nominal radius plus 0.03-m reserve) enters unknown 5-cm floor
cells beyond those already intersected by the current footprint and predicted
hold path. The hold comparison includes the common committed prefix. Existing
unknown cells are not declared free or safe by this comparison.

`scripts/probe_go2_recorded_footprint_extension_development.py` evaluated all
160 saved planning states through frame 640. The signal would change 34 selected
actions: 25 translations and nine pure left turns. Every changed state retains
at least one candidate passing the original clearance/stopping checks and the
additional coverage comparison, although several replacements are holds.
This evaluates saved alternatives, not their resulting trajectories or runtime
latch transitions.

Translations are flagged at frames 116–152 and again at 452–560, before the
first sampled physical margin crossing at frame 576. This provides a potential
approach-time signal. However, applying it indiscriminately also replaces an
initial survey turn with hold at frame 16, while coverage is still sparse.
A blanket rule can therefore obstruct the observations needed to explore.
Neither collision avoidance nor mission completion is established by this probe.

Evidence: `recorded_footprint_extension_all_plans_v1.json`; the smaller initial
sample is `observed_floor_footprint_extension_probe_v1.json`.

## Next experimental change

Test translation-specific footprint coverage paired with an explicit request
to observe the missing region. Keep pure-turn clearance rules, six candidates,
model weights, nominal/reserve margins, stopping checks and measured dispatch
guards unchanged. A coverage-directed view request should resolve through
actual floor or obstacle observations; reaching a heading alone must not mark
unknown cells as observed. Weak-view tracking recovery remains relevant and
must not be silently bypassed. Do not use simulator wall thickness or hidden
wall geometry in the controller.

Before a full mission, check the new rule on recorded states for its intended
activation and preserve the distinction between unavoidable committed motion
and action-dependent footprint extension. Then evaluate a full exposed-layout
mission, including stuck inspections and tracking failures, before any fresh
replication. Neither the arc fallback nor disabled heading release is a proven
repair. Broader reliability, learned-model contribution and realistic sensing
remain unresolved.

The completed failure recordings currently leave approximately 0.5 GiB free.
Further native collection needs routine eligible depth retirement first; the
map/coverage analyses above fit without modifying or retiring any failure.
