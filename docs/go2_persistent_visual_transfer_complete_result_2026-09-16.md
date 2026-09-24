# Four fresh-maze pairs: complete result

The fixed learned controller and fitted pose/command control each completed
four physically verified goal-and-return missions on four fresh development
layouts, with zero disallowed contacts. All eight planned outcomes are included.
The controller, trained models, layout inventory, sensing and arrival rules
were fixed before this batch; no navigation outcome changed them.

| Development layout | Learned seconds | Pose/command seconds | Round trips | Contacts |
| --- | ---: | ---: | --- | ---: |
| 0 | 160.92 | 445.28 | Both passed | 0 |
| 1 | 204.88 | 199.28 | Both passed | 0 |
| 2 | 148.42 | 176.06 | Both passed | 0 |
| 3 | 201.68 | 237.22 | Both passed | 0 |

Every goal and home arrival passed the unchanged physical requirement: within
40 mm for one second, zero requested commands and measured 100-ms speed below
50 mm/s. The learned model's raw nominal-composed forecasts actually selected
and checked candidate actions, without external neural motion correction.
The control used fitted pose/command XY and command yaw; both arms computed
the same forecast alternatives and retained identical perception and safeguards.
Native geometry and physics were used only for evaluation.

Mean/median mission duration was 178.975/181.30 s for learned and
264.460/218.25 s for pose/command. Mean paths were 15.683 and 16.147 m.
These timing differences do not establish forecast superiority: layout 0's
control missed 690 of 693 planning deadlines during its long return pause.
The pose/command predictor also had lower same-window XY forecast RMSE than
the neural model on all eight executed trajectories. Those 700-ms windows
overlap and are diagnostic, not independent missions or counterfactual policies.

This is preliminary generalization within one procedural maze family, with one
execution per condition/layout and one training seed. It does not establish a
reliable deployment success rate, learned/RGB/JEPA superiority, isolated online
prediction or memory benefits. Sensors still use synthetic 2-mm depth noise and
ideal gyro; measured simulation incurred substantial lag. No real-time or
hardware validation is claimed. Earlier development failures remain preserved.

Aggregate: `go2_persistent_visual_transfer_complete_v1_attempt_001/result.json`.
Four paired results and PNG/SVG figures use
`go2_persistent_visual_transfer_comparison_layout00_v1_attempt_001` through
`go2_persistent_visual_transfer_comparison_layout03_v1_attempt_001`.
Execution journal: `go2_persistent_visual_transfer_2026-09-16.md`.
All four paired PNGs were visually inspected. Completed depth recordings were
retired under the standing policy; all non-depth evidence remains available.

Next: compare the already-trained, frozen JEPA and direct-prediction models on
these same locked layouts with the unchanged controller. Then address matched
reactive/prediction-off controls, replication, sensor/timing realism and hardware
evidence. The broad navigation goal remains incomplete.
