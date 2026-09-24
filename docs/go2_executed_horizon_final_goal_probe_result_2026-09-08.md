# Executed-horizon final-goal native result: no improvement

Both fixed corrected models completed collection and raw audits with all
measurement gates passing. **0/2 verified arrivals.** JEPA's shorter final-goal
pose score worsened closest and terminal distance relative to the exact-target
predecessor. Do not adopt it as an arrival improvement.

| Condition | Native closest distance | Native terminal distance | Terminal |
| --- | ---: | ---: | --- |
| JEPA | 0.06736172244572997 m | 0.07060121777053317 m | Mission tick budget exhausted, tick 243 |
| Direct | 1.1011647990168776 m | 1.1011647990168776 m | No feasible phase candidate, tick 46 |

The JEPA predecessor reached 0.03971029335649642 m and ended at
0.042645782907013687 m, also without satisfying the complete arrival gate.
The new JEPA run failed even the evaluator-only 6-cm one-second quiet test.
It acquired 254 paired frames over 253 commands/13,400 physics samples. Direct
remained identical over 57 frames, 56 commands and 3,550 physics samples. Both
retained the ten-command terminal drain. Neither had physical/acquisition
stops. No thresholds, budgets, contacts or nominal constraints changed.

The completed readout verified JEPA's first command change at 171 and exact
172-frame native/public/RGB/auxiliary/map/forecast prefix. The complete direct
comparison was unchanged. Between the two new models, command difference 26
bounded the exact 27-frame raw/public/RGB/auxiliary/map prefix; forecasts differed.
JEPA changed 69 selections relative to its intermediate eight-step action under
the new score. No active infeasible waits occurred for JEPA; direct retained
ticks 36–45. Model states remained unchanged and full raw command replays passed.

All 311 paired auxiliary frames passed strict visibility with zero robot pixels.
Maximum observed XY errors were 2.2104 mm JEPA and 1.3957 mm direct. JEPA median
complete command iteration was 819.298567 ms (maximum 1180.411417), direct
818.1556245 ms (maximum 1064.147853). Every command iteration exceeded 100 ms;
acquisition medians alone were 207.891237 and 200.390518 ms. Native computation
plus audits took 430.1334059089422 s. No real-time qualification is established.

An initial read-only executed-prefix check found 72 actual JEPA final-goal
steps with mean 100-ms XY error 0.007810341629022737 m and maximum
0.02488901046179946 m. Mean predicted-minus-native residual was
[-0.0005565873668591931, 0.005060518556182659] m. Several late predictions had
positive body-y movement while the actual step had negative body-y movement.
The predecessor's 71 final-goal steps had mean error 0.009148377082264857 m.
Lower average prediction error here did not produce better navigation. These
post-hoc native labels must not feed an online correction. A separate fixed
prequential diagnostic will check only past public-observed residuals.

Under the approved artifact base:

- `go2_executed_horizon_final_goal_probe_v1_attempt_001/launch.json`:
  `6327f7d8ed5cafe8a4f526aa273313c3c71a84e67eed252186b80bb5670400f7`
- Native result:
  `dabc90b19f1d8285d21fd81d265efae504fbe75b1868357f6bcd17f2e77fe062`
- `go2_executed_horizon_final_goal_readout_v1_attempt_001/launch.json`:
  `3049b0907dac774803dde0e2b806e861538942c5424f3dfd8b8ea662ff7c0b7c`
- Readout result:
  `edd050780b55f45956cb685d57b989cb748ed90227bc8937bec088589d755559`

Native/readout closures contain 1,344/1,347 paths. Original attempts remain
immutable. This reused development layout supplies no independent-maze,
backtracking, baseline or hardware qualification. The full goal remains open.
