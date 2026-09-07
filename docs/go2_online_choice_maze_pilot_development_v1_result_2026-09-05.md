# Fresh online conditional-choice pilot: completed and audited

The actual current-RGB/body-packet → fixed ensemble → selected command → Go2
physics path has now been executed on eight new topology-disjoint layouts.
All 72 prescribed trials completed; every prefix was available. The full audit
passed raw physics, native contact attribution, sensor/camera histories, actual
packet/ensemble replay, requested/post-slew tapes, utility and paired reduction.

**The result does not establish useful JEPA navigation.** Supervised rollout
without latent prediction was contact-free but only weakly better than stopping
on average. JEPA was worse than stopping and incurred two contacts in one layout.
This is a conditional single-decision result, not maze exploration or return.

## Fixed-method results

Twenty-four local decisions per method: eight independent layout draws, each
with three supplied body-frame intents. The three intents are not independent
maze replications. Lower cost is better; contact/init/incomplete/release failure
costs 10, otherwise cost is the measured 4-s displacement error.

| Method | Mean realized cost | Native contact stops | Selected stop |
|---|---:|---:|---:|
| Always stop | 0.7994 | 0/24 | 24/24 |
| Supervised-rollout ensemble | 0.7077 | 0/24 | 6/24 |
| JEPA-rollout ensemble | 1.7151 | 2/24 | 10/24 |

Supervised-rollout minus stop cost: −0.0918, descriptive layout-bootstrap
95% interval [−0.2341, +0.0827]. Six layouts favor supervised rollout; two favor
stopping. The interval crosses zero, so this small development pilot does not
establish a reliable improvement over stopping.

JEPA minus stop: +0.9157, interval [+0.1243, +2.4241]. JEPA minus supervised
rollout: +1.0074, interval [+0.1200, +2.5875]. The large adverse cost includes two
reverse-action contacts in layout 03; these are correlated failures, not two
independent maze replications. All original negative outcomes remain retained.

The methods use a fixed three-seed prediction ensemble. They are not the same
policies as individual-seed rows from the earlier offline diagnostic, and the
primary utility here additionally penalizes release/init failure. Do not compare
the two studies as if only the environment changed.

## An actionable model limitation

The supervised model sometimes moves away from the supplied intent because of
its predicted action costs, not because the adapter changes the selected action.
For example, layout 01 / left intent:

- Predicted stop probability 0.1204 made stop cost 1.937, despite predicted
  displacement error only about 0.733 m.
- Predicted forward probability 0.00141 made forward cost 1.561, so it selected
  forward rather than stop or the left curve.
- The actual stop comparator was contact-free with cost 0.7964; actual forward
  cost was 1.4548 and moved away from the sideways target.

Exact selection replay and raw command auditing confirm this chain. This points
to risk/readout quality and the decision objective as next diagnostic targets;
it is not evidence for flipping action signs or changing the sensor frame.
An individual predicted probability cannot be declared globally miscalibrated
from one realized outcome, but the paired cost consequence here is directly
observed. Zero measured contacts alone would hide the failed sideways intents.

Do not change these completed models, action costs or trial population. A future
fixed diagnostic can separate the accurate training-action mean dynamics from
the learned scene-risk readout, using training-only motion information. That
would be a new method, not a repair retroactively attributed to this pilot.

## Scope, timing and evidence

The audit covers 333,400 physics samples, 33,340 ideal simulated body samples and
5,660 RGB/history packets. Every physical/body prefix and camera geometry matched;
the declared sparse RGB variation limits held. Each method used its **own actual
current image**, never a substituted sibling/canonical observation. All selected
model outputs were reproduced exactly during the final audit.

Median learned adapter wall time was 10.7 ms (supervised) / 10.6 ms (JEPA);
median capture-plus-adapter was 21.6 / 21.5 ms. The synchronous simulator paused
during that computation. These measurements do not validate real sensor delays,
robot scheduling, hardware inference or closed-loop real-time safety.

Teacher initialization still used privileged pose. Contact/stability emergency
stops were privileged simulator safeguards. The observation regime remains ideal,
the maze materials/style and single local approach are narrow, and the policy
makes one open-loop choice. No persistent place memory, hidden-beacon exploration,
return, repeated replanning or physical transfer was evaluated.

Output: `.generated/go2_online_choice_maze_pilot_development_v1_attempt_001`.
Collection 61885 and full audit 63999 both terminated with exit 0.

- Result SHA-256: `c8e25c20f52d03a6bc165757c2ebcdad1b620d996a72bb95610f7f5bde9c8045`.
- Full audit SHA-256: `ed652233fd49c52e023bbd01d11d5ffafad155a8f2992d9adc4740212287e7a0`.
- The earlier 39-trial interim audit is retained but superseded by the full audit.

Next: qualify sensor-based look/turn-back primitives, broaden causal temporal and
decision-state coverage, retain the non-JEPA supervised baseline, and integrate
online place/exit memory only with honest association/execution evidence. The
[gyro-turn preparation](go2_relative_gyro_turn_development_preparation_2026-09-05.md)
is implemented and synthetically tested, not yet physically qualified.
