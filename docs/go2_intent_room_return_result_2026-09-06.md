# Multi-reference/persistent-intent room return: audited result

The fresh batch finishes at **0/3 full returns under the preregistered criteria**.
It does establish a physically executed final-home hold in the left trial,
but one intermediate hold fails. Preserve that distinction: neither a
controller candidate nor a passing final endpoint erases a required local failure.
The ultimate novel-maze JEPA goal remains unachieved.

## Evidence

Output root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_intent_room_return_v1_attempt_001`.

- Launch SHA256: `7a5c427ca521de367a301376fafd262aeda5f6e16b7ed876f2b40e87ce0b1a91`.
- Collection result: `27e2f91eaece8667e48fb98d75ce3ee8cfcc3a1c9d0052a97033cad5acc321d3`.
- Independent audit launch: `149541ff0a1bf1f533d4b565f4431b7132fa0e3303e640095d23288e50a3baa7`.
- Independent audit result: `a350c74d4f5851a7486bf01a8276be7f9eb420a83cd8198ad684159b8385923d`.

All expected artifacts are present. The independent audit exactly reconstructs
4685 decisions and their stored return intent,4702 commanded intervals and4705
RGB-D observations. All depth checks are within1 mm. There are237350 native
physics samples,474.7 simulated seconds and173 pulses. No native physical stop
or storage stop occurred. Source and separate external-data bindings pass.

| Trial | Stages | Native pose holds | Signed-winding holds | Outcome |
| --- | ---: | ---: | ---: | --- |
| Nominal left | 7/7 | 7/8 | 8/8 | Final home passes; intermediate leg6 fails |
| Nominal right | 6/7 | 7/7 | 7/7 | Tracking fails at frame1809 |
| Lower-friction left | 0/7 | No completed holds | No completed holds | Planned endpoint excursion stop at tick461 |

Extra clipped legs explain why stages and holds are different populations.
Overall14/15 pose holds and15/15 winding holds pass. There is no matched causal
comparison with older changed-start/appearance batches.

### Left: actual home, insufficient intermediate margin

The controller reaches a return candidate at tick2412. The final native1 s
home hold has maximum position error0.0536378956 m, heading0.0431892202 rad,
speed0.0004907393 m/s and yaw rate0.0074436540 rad/s; all pass unchanged limits.
But clipped-home leg6 reaches0.0615571014 m maximum native position error,
exceeding0.06 m. Its other criteria and signed winding pass. No rounding or
extra tolerance changes this failure. Maximum available visual position error
over the trial is0.0096301978 m. Native path length3.9025166864 m;90 pulses.

The sensor and native acceptance regions cannot safely be treated as identical
when pose error is nonzero. An internal control margin is a plausible next
engineering change; it is not yet a validated uncertainty bound.

### Right: multi-reference support is still insufficient

No retained reference qualifies at frame1809. Reconstructing the active
reference1787 gives24/48 inliers (0.5, below0.6),6 reference grid cells and5
current cells (below6), with only0.0031968 m reference translation. Both
consensus fraction and support fail; this is not just a translation excursion.
The other seven retained references also fail unchanged gates. See the
[read-only diagnostic](go2_intent_return_visual_failure_diagnostic_2026-09-06.json).
Maximum available visual position error0.0112753352 m; path2.9817354058 m;
65 pulses. No final home hold is declared.

Neither nominal nor low-friction stream used a successful alternative-reference
fallback in this fresh batch. The earlier two replay fallbacks do not establish
fresh closed-loop benefit from the buffer. Do not claim that enlarging the
buffer or lowering the inlier/grid gates would fix this failure.

### Low friction: state-independent dynamics transfer fails

Tracking remains available through the controller stop. The first goal is not
reached; after18 pulses a proposed next endpoint exceeds the existing excursion
limit. At the last decision, visual position is approximately(.46969,.18342)
and yaw-1.31477 rad relative to departure, versus goal(.4,0)/yaw0. Native final
position is(.47301,.18187), yaw-1.31989 rad. Maximum visual error0.0042757814 m;
path0.9993515619 m. Good tracking alone does not make the nominal action table
transfer to changed support dynamics.

## Timing and scientific scope

Median observation/control wall times are146.50,142.55 and144.72 ms; p95 values
155.20,150.74 and149.90 ms. They exceed the nominal100 ms control interval.
Physics is paused for computation, so this is not realtime execution. At least
159.60 GiB remained free during decisions on the new filesystem; the40 GiB
reserve was retained, and no old evidence was moved or deleted.

The controller remains hand-engineered above a learned gait. Scripted metric
waypoints are not a demonstrated exploration/memory advantage. Ideal hidden-robot
RGB-D and body sensors, controlled floor, uncalibrated uncertainty and no hardware
remain limitations. The separate [pulse target/loss work](go2_pulse_timed_target_and_loss_progress_2026-09-06.md)
now joins actual recorded observations and native labels to matched untrained
objectives; it does not yet establish learned dynamics or useful JEPA prediction.

Continue with the [execution-to-learning plan](go2_intent_return_to_jepa_next_steps_2026-09-06.md).
