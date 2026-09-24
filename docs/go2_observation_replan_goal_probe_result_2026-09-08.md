# Observation-cadence replanning native result

Both fixed augmented first-seed cases completed and passed full raw sensor,
model/command replay and strict physical visibility audits. No physical or
acquisition stop occurred, and there were no hard measurement failures.
Neither case reached the goal. The unchanged goal gates remain in force.

Full direct stopped at tick 29 on the nominal/surface candidate constraint,
with 39 commands including terminal drain, 40 RGB-D frames and 2,700 physics
samples. Minimum/terminal actual goal distance was 1.097549739 m. Its 27
selections were 18 right turns, seven right arcs, one hold and one no-action
terminal. Replanning more frequently did not prevent its clearance failure.

Full JEPA exhausted the mission budget at tick 243, with 253 commands,
254 RGB-D frames and 13,400 physics samples. Minimum/terminal actual goal
distance was 1.175892652 m. All 240 selections stayed in view acquisition:
14 right turns and 226 holds. At late tick 242 the remaining scan error was
0.111246 rad, outside the unchanged 0.1-rad scan transition tolerance. Its
half-second hold forecast claimed 0.065739 rad of heading improvement and
outscored the turn, leaving the selector stalled. This motivates testing
predictions at the actually executed interval; it does not prove that a
shorter model horizon will solve the problem.

Each model matches its own five-command predecessor's raw/public/RGB and
observer/memory prefix through the observation before the first different
command: 18 frames for JEPA (first difference tick 17), 20 for direct
(tick 19). The two new cases match through 18 frames. Maximum observed-pose
XY error was 2.107 mm for both. There are still zero verified arrivals and
zero independent novel mazes.

The two-case phase took 287.933 seconds after launch. Maximum worker RSS was
2,534,297,600 bytes. Preflight observed 82,328,260,608 bytes available RAM and
70,758,895,616 bytes artifact space. Complete-iteration medians/maxima were
513.846/776.144 ms for JEPA and 500.742/558.176 ms for direct; every complete
iteration exceeded 100 ms. Native physics was paused during compute, and
these measurements do not establish real-time or hardware operation.

Exact identities under the established navigation development artifact root:

- `go2_observation_replan_goal_probe_v1_attempt_001/launch.json`:
  `8ae2467354cd0c040e13e7c346bca1cf989e8e32783dc599e1151994cbba97c2`.
- Its `result.json`:
  `8cdfd800eda961de5b1a58a3b64fe8fd471c0f8c8165aae9c61f699c500010eb`.
- `go2_observation_replan_goal_readout_v1_attempt_001/launch.json`:
  `5659ef229974d7bdd7c8bb81f2b34dc02c881e963a411fff401e0719b357aeb3`.
- Its `result.json`:
  `be2c24798bca35111c62b3795c42c8eb14783405b7b1e6afe3b2f7a1c59f554d`.

The probe binds 1,094 source paths and 1,242 artifacts; readout binds 1,097
source paths. Three focused controller tests and one readout-boundary test
passed. Collector changes were checked as restricted to controller type and
labels; the audit body changed only controller type and uses the original
native goal and actuator auditor objects. Earlier results are unchanged.
