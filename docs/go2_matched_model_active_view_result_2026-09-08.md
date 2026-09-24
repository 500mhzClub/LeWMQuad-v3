# Matched-model active-view native result — 2026-09-08

All twelve fresh cases completed raw audits, and none reached its mission goal.
No physical contact or strict/hard depth failure was recorded. Changing the
fitted predictor improved some forecasts and changed actual behavior, but did
not produce successful navigation under the shared waypoint controller.

The [protocol](go2_matched_model_active_view_probe_v1_2026-09-08.md) assigned each
of the six existing final fits to both known mirrored layouts. Each case used a
fresh process, scene, model reload, observer and persistent map. Model/head/input
treatment changed; the scan cost, waypoint rules, six actions, five-tick
commitment, 1.2-m goal, 240-tick budget, physical guards and native goal audit
were fixed. No-RGB removed RGB from the predictor, while the shared visual
observer and mapper continued using it. This is not whole-system RGB removal.

| Model | 052 terminal / minimum goal distance (m) | 052 terminal | 039 terminal / minimum goal distance (m) | 039 terminal |
|---|---:|---|---:|---|
| RGB direct | 1.088751 / 1.087257 | Tracking failure at tick 98 | 1.186206 / 1.152475 | Tick budget |
| RGB supervised rollout | 1.084514 / 1.076638 | Tracking failure at tick 119 | 1.097864 / 1.082719 | Tracking failure at tick 100 |
| RGB JEPA | 0.973902 / 0.973902 | Tracking failure at tick 232 | 1.185256 / 1.184504 | Tick budget |
| No-RGB direct | 1.213360 / 1.152812 | Tick budget | 1.163020 / 1.132861 | Tracking failure at tick 193 |
| No-RGB supervised rollout | 1.224805 / 1.192683 | Tick budget | 1.192011 / 1.144332 | Tick budget |
| No-RGB JEPA | 1.183625 / 1.181223 | Tracking failure at tick 55 | 1.185724 / 1.152475 | Tick budget |

Eleven cases entered waypoint mode; RGB JEPA on 039 stayed in scanning. Most
selected actions were in-place turns or holds. For example, RGB direct on 039
made 42 waypoint forecasts but selected only 16 right turns and 32 holds across
the whole run. No-RGB JEPA on 039 also made 42 waypoint forecasts; 23 selected
hold, 15 right turn and four left turn. Neither moved meaningfully toward the
goal. Nonzero waypoint command counts therefore must not be presented as forward
travel or navigation success.

Source inspection shows that the shared waypoint score values predicted XY
displacement and four-second contact cost, without explicit reward for turning
toward a sideways waypoint. Its contact horizon also exceeds the next five
committed commands. These limitations motivate the separately specified
[commitment-pose utility](go2_commitment_pose_waypoint_utility_v1_2026-09-08.md),
which changes both waypoint value and scoring horizon. The present cohort was
not modified to test that revision. Model forecast error remains unresolved.

Six runs exhausted their navigation budget and six stopped on unavailable visual
pose. RGB JEPA 052 exhausted its bounded measured bridge despite an available
incremental measurement. The other five tracking failures had neither accepted
anchor nor incremental pose. No-RGB JEPA 052 lacked enough rigid-pose matches;
the other four rejected rigid consensus fraction, grid support or displacement.
The largest accepted-pose XY error was 6.026 mm. This does not provide a pose
after failure or a calibrated error bound.

The [readout](go2_matched_model_active_view_readout_v1_2026-09-08.md) verified that
both new full-RGB JEPA cases exactly reproduce their original active-view runs:
all command tapes, RGB hash sequences, physics/contact arrays, public body/control
histories and fast-gyro arrays match. Every decision also matches after removing
only the changed controller label and added model-condition/input-variant fields.
The original failures are preserved. ZIP timestamps and wall timing are excluded
from this explicit numerical comparison.

Nine focused tests passed before launch. All twelve cases passed complete raw
sensor-to-observer-to-map-to-model-to-command replay with unchanged fitted model
state. The cohort retained 2,387 camera frames and 127,750 physics samples; all
2,387 recorded strict/hard depth checks passed. Every complete control iteration
exceeded 100 ms. Per-case median iteration times ranged from 281.744 to 309.508 ms,
with a maximum of 649.049 ms. Physics paused during computation, so these runs
do not establish real-time execution. The serial cohort took 1,689.946 s after
launch, including workers and audits.

Hardware was inspected before launch and monitored throughout. Approximately
82 GB RAM and 95 GB artifact space were available initially; serial fresh workers
preserved uncontended timing. There was no substantial competing compute job.
The full objective remains unfulfilled: this is a failed known-layout comparison,
not independent-maze evaluation, useful physical backtracking, calibrated safety
or real-platform qualification. No model has been promoted as a successful
navigator. Subsequent development can use the RGB direct fit as a lower-motion-
error reference alongside the original RGB JEPA fit, with that choice declared.

Artifacts use the existing development base. Native root:
`go2_matched_model_active_view_probe_v1_attempt_001` (924 frozen sources; 9,934
bound artifacts totaling 4,166,141,598 bytes). Readout root:
`go2_matched_model_active_view_readout_v1_attempt_001` (926 frozen sources).

| Identity | SHA-256 |
|---|---|
| Native launch | `bf03f9887ca55a2b874869947b2ebdd4d77589cfdcb7b39696b126833fc30cc2` |
| Native result | `4a921e6856db9dc0e8033fe9e4768126ceaf90b402cf195f91bcb3e3c43384e1` |
| Readout launch | `df8dfd250797dd1adeac6f271bbae0ee13357025b076886e5bf4c79f5c87c2b7` |
| Readout result | `13a22594165199a16a4b0f42c1d3e983e1dea9e13cd13d4b03707c371b7d2320` |

Every source/input/native/model/artifact binding was reauthenticated. All cohort
processes are terminal. No case was resumed, replaced or omitted.
