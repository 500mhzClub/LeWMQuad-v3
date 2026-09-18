# Completed six-model development batch

All six preassigned models completed collection and full raw audits on reused
development maze 02. The original parent ended with
`ALL_PHASE_ADAPTER_MAZE02_MATCHED_NATIVE_V1_COMPLETE`. There were **zero verified
round trips**. Only the no-RGB direct-head model reached the outbound goal.
No model, policy, or independent-study budget was selected from these results.

| Model input / training head | Distinct outbound edges | Outbound arrivals | Round trip | Strict visibility | Median observation/control |
| --- | ---: | ---: | --- | --- | ---: |
| Full / JEPA | 2 | 0 | Failed | Failed | 1,990 ms |
| Full / supervised rollout | 0 | 0 | Failed | Passed | 2,699 ms |
| Full / direct | 0 | 0 | Failed | Failed | 2,581 ms |
| No RGB / JEPA | 5 | 0 | Failed | Passed | 1,024 ms |
| No RGB / supervised rollout | 0 | 0 | Failed | Passed | 2,672 ms |
| No RGB / direct | 10 | 1 | Failed | Passed | 1,442 ms |

The full JEPA run terminated with no phase candidate satisfying surface and
nominal constraints. Its audit retained hard measurement failure frame 1173.
The no-RGB JEPA run stopped on a sensor/model failure; the previously diagnosed
tracking issue is addressed by a queued diagnostic. The other four runs exhausted
the shared mission budget. Both supervised-rollout variants remained in the
starting cell. All six had zero recorded native physics contacts and no invalid
cell-edge crossings; this does not prove collision clearance.

The no-RGB direct run reached the outbound goal at frame 2935. Its native
one-second arrival window passed, with maximum goal distance 0.0346 m and maximum
speed 0.0256 m/s. It then entered RETURN but crossed no return edges before the
deadline. The existing [return-budget diagnosis](go2_no_rgb_direct_maze02_return_budget_diagnosis_2026-09-11.json)
shows 6.8 seconds remained versus a 15.35-second unobstructed travel lower bound.
The already queued 4,000-tick diagnostic will test the original controller with
more time; it does not retrospectively change this failed run.

All 14,455 observation/control samples exceeded the 100 ms command interval.
Simulation physics was paused during computation. These timings neither measure
nor qualify continuous real-platform control. The separately running receipt-copy
replay is a controller-only optimization comparison on an older fixed history.

The direct head predicts action-conditioned future outcomes; it is not a
non-predictive baseline. Removing RGB from the learned model retains the original
RGBD controller frontend. This one-seed, one-reused-layout batch establishes no
JEPA advantage, RGB dispensability, causal planning or memory benefit, or
independent-layout reliability. The independent study must retain appropriately
labelled reactive and memory comparisons after the five diagnostic stages and
final policy/budget review.

## Evidence and verification scope

- Batch result SHA-256:
  `9715a916b81d9a70edf2d26e4f3d7e952823d3e6b618670c46ab2aa0429ff416`.
- Launch SHA-256:
  `97c307ce8d2178494e7484f14b3b0cdcd8d7d0ceaab2f94d94d59da44e39b17a`.
- [Bound readout verification](go2_completed_adapter_batch_readout_verification_2026-09-11.json):
  `4a7589c46155f091c756893083282159504656e07a37e4371d0762851f09685f`.
- Reproducible checker: `scripts/read_go2_completed_adapter_batch_v1.py`;
  session 29776 exited zero.

The checker verified the original 1,908 source bindings within a 1,916-path
checker closure. It rehashed 51 selected artifacts before and after: the final
result/launch, resource monitor, and each worker's terminal, audit, startup,
readout, parent acceptance, log, collection result and physics archive. It
reconstructed every readout from the bound audit/collection and complete native
contact arrays, reran the original six-case completion checks, and reconstructed
the exact 86,966-entry final artifact roster from worker records. The six contact
arrays cover 726,950 physics samples.

This readout check did not rerun RGBD/model auditing, rehash every generated
artifact, or repeat full training/runtime input admission. Sensor/model and
visibility findings come from the authenticated completed raw audits. The native
frontier diagnostic's original admission path performs its own full input checks.
