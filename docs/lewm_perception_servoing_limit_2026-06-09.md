# Pure-Perception Multi-Beacon: Recognition Works, Approach Doesn't

Date: 2026-06-09

A closed-loop test of whether the LeWM substrate can drive **pure-perception**
navigation between beacons — image-goal servo to a beacon, switch to the next on
arrival, and **scan (rotate) to find the goal beacon when it is not in view** —
with **no privileged breadcrumb waypoints**. Reproduce with
`benchmark_lewm_closed_loop_mpc.py --demo-perception` (the only privileged inputs
are the per-beacon goal keyframe and the claim radius).

## Result: it cannot navigate by perception alone

The robot can **find** a beacon but cannot **approach** it. Two image-based cost
models were tested in closed loop; neither provides a usable homing gradient.

### plan_cost (latent L2) — recognition only, flat over distance

Starting facing away from a beacon 1.55 m behind it, the scan (`yaw_left`)
lowers the cost 530 → 348 as the beacon rotates into view (recognition works).
But once it commits to `forward`, the cost stays **flat at ~347** while the true
distance climbs 1.55 → 6.3 m — it walks straight past/away:

```
d=1.55 best=502 hold=530  -> yaw_left    (scanning, beacon entering view)
d=1.55 best=348 hold=371  -> forward_fast
d=2.10 best=347 hold=370  -> forward_fast   (cost flat, walking away)
d=6.33 best=348 hold=371  -> forward_fast
```

`plan_cost` rewards "beacon in view," not "beacon centered / closer." There is
no approach or centering gradient; the single-beacon demo only succeeds because
it *starts pointed straight at the beacon*.

### RelPoseHead (metric) — confidently miscalibrated

Wiring the pose-aux ladder's `RelPoseHead` (cost = predicted `‖dxy‖`-to-goal) as
the servo signal gives a strong but wrong signal. Its predicted distance is
~0.2 m essentially constant, regardless of the true distance:

| true distance | pose-head predicted ‖dxy‖ |
|---|---|
| 1.55 m | 0.18 m |
| 1.88 m | 0.32 m |
| 4.18 m | 0.20 m |
| 5.06 m | 0.20 m |

It does not track distance, so the robot follows a confident (sig 0.5–0.8) but
meaningless gradient and drifts away. This matches the nav-cost diagnosis that
the head cost is only *weakly* metric (ρ≈+0.22 single / +0.38 multiview).

## Conclusion

Neither image-based cost gives a usable homing gradient: `plan_cost` is flat
(recognition), the pose head is miscalibrated (too weakly metric). The robot can
*recognize* a beacon's presence (the scan works) but cannot *approach* one by
perception. This is the project's foundational result — **the LeWM latent is a
place-recognition code, not a metric code** (Phase-A: ρ≈0.03 for the latent;
[[project_lewm_aliasing_a2]], [[project_lewm_nav_cost_phase0]]) — now confirmed
**in closed loop with two distinct cost models**.

Therefore a genuinely pure-perception long-range traversal is **not reachable
with this substrate**. It needs the topological-nav **reachability / subgoal stack**,
which turns a long approach into recognition-sized hops between learned
subgoals — exactly the role the privileged breadcrumb waypoints were standing in
for in the multi-beacon *demo*. The breadcrumb 2/2 demo is therefore labeled
"image-goal servoing + privileged subgoals (H-JEPA placeholder)", not pure
perception.

The servo+scan logic and both cost paths are retained behind
`--demo-perception` for re-running this negative result.
