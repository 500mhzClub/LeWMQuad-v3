# V3 recovery-support diagnosis and V4 decision

Both bounded command-prefix replays passed: 2,301 frame pairs/poses for01 and3,301 for09. Every consumed RGB-D hash and unchanged raw pose matches; all native states match the source prefixes exactly. No controller was rerun, no feature/model fitting occurred, and no frames were retained. The two replays used563 simulated seconds including settling,806.1 wall seconds, and13.57 MB of retained diagnostics, within the frozen limits.

| Inspected window | Frames | Original strong-corner maximum below48 | Actual tracker maximum on those frames | Actual maximum below48 |
|---|---:|---:|---:|---:|
| 01,190–230s |401|98|85–150|0|
| 09,267–330s |631|99|150|0|

Examples:01 at190s has original strong counts35/24 but actual selected tracker counts125/150.09 at270s has40/46 versus150/150. Both frames have an admitted unchanged visual pose and a logged weak-support recovery objective. The tracker already fills its sparse feature budget with additional measured weaker corners; recovery deliberately continues to inspect only the older stronger subset. These counts are not calibrated tracking confidence. Exact replay proves the retained-input relationship, not that all subsequent motion through such views will remain trackable.

V3's latch release removes all705 stale-memory holds in01 but the resulting motion repeatedly alternates route turns and measured-view recovery. In09 it brings beacon arrival forward to266.8s, then the return becomes weak-view recovery with blocked movement. The remaining common mechanism is unnecessarily retreating to a historical view based on a feature subset smaller than the one used by the accepted tracker.

## Single V4 change

Use the unchanged tracker's actual selected feature counts, including its already-existing sparse corner completion, as the input to the existing visual-support recovery policy. Retain the48/96 thresholds, both camera witnesses, same accepted poses, existing V2 one-second attained-reference exhaustion, V3 blocked-latch release, routing, safety/dispatch checks, models and candidate bank. Missing witnesses remain unavailable; genuinely sparse actual support still requests recovery.

This changes recovery behaviour only. It does not change feature extraction, corner selection, matching, pose fitting, admission thresholds or estimator state. No native pose/geometry enters the deployed policy. Preserve original strong counts in each recovery receipt alongside actual counts, and retain scalar visual-support receipts with pose logs for subsequent diagnosis.

Expected effect: fewer spurious turn reversals while the existing tracker still admits adequately populated feature sets. Risk: the stronger-corner subset may have warned of impending pose loss, so using the larger set can miss that warning. Navigation outcomes and the unchanged oracle gate must test this; selected feature count alone does not certify safe tracking. Any disallowed contact or hard violation stops the version under the existing rule.

Four focused pre-run checks cover the count substitution without mutating the raw pose/witnesses, missing evidence, genuinely sparse-support recovery, and construction preserving the underlying registration, recovery state and publisher. V4 consumes the fifth of six outcome versions. Freeze first, re-project time/storage, then run the original ten-episode C1 screen; if9/10, run the ten second episodes requiring9/10 before C0. No second-episode or oracle gate has yet run.

Evidence: `v3_support_replay_attempt001` under the programme artifact root. Frozen plan: `docs/go2_navigation_capability_v3_support_replay_plan_2026-09-27.json`. All original failures, source recordings and the constructor erratum remain preserved.
