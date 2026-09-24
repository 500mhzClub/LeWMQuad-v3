# Successive RGB/body choices: completed physical result

All144 fixed trials completed and passed the corrected full physical/decision
audit. The experiment provides a **bounded positive result for the JEPA-trained
half-second latent-prediction head**, alongside substantial remaining failures.
It does not establish safe novel-maze exploration or a benefit from multi-step
latent planning.

## Fixed comparison and outcomes

The [protocol](go2_successive_choice_maze_development_v1_2026-09-05.md) used eight
fresh topology-disjoint procedural layouts, three initial-frame directions and
six methods. Each learned method averages its three fixed final training seeds.
After a common teacher initialization, actual RGB/body/control history drives
eight possible half-second decisions, followed by a half-second zero release.
All144 prefixes were available; there were no sensor-contract failures. There
were45 native contact stops across all methods. No physical trial was rerun.

Every row below has24 trials. “Complete” means the full command duration and
release were executed, not that a beacon, maze goal or viable route was reached.

| Method / inference head | Contact trials | Complete control + release | Release-motion pass | Mean signed displacement to observed endpoint, m | Mean four-second displacement among completed trials, m |
|---|---:|---:|---:|---:|---:|
| Always stop | 0 | 24 | 24 | .0006 | .0006 (n24) |
| Direct-only / direct | 7 | 17 | 17 | .5347 | .6010 (n17) |
| Supervised auxiliary / direct | 9 | 15 | 13 | .6196 | .7524 (n15) |
| Supervised auxiliary / latent head | 12 | 12 | 10 | .7046 | .9159 (n12) |
| JEPA / direct | 13 | 11 | 8 | .5983 | .8360 (n11) |
| JEPA / latent head | 4 | 20 | 18 | .6477 | .6703 (n20) |

Observed-endpoint displacement includes motion before early contact and therefore
mixes durations. Completed-trial means have different surviving populations;
they must not be ranked as an unbiased all-trial performance comparison. The
machine-readable result also retains lateral displacement, observed duration,
all missing fixed-horizon outcomes and the individual physical trials.

## What this identifies—and what it does not

For JEPA latent head minus the matched supervised latent head, contact incidence
changes by−33.3 percentage points: descriptive paired-layout bootstrap95%
interval[−50.0,−16.7]. Contact incidence is lower in six layouts and equal in two.
Completion and release-pass differences are both+33.3 points, with intervals
[+16.7,+50.0]. The matched completed-intent progress comparison omits12 of24
pairs, explicitly reported in the result. It cannot establish general all-trial
progress superiority.

Within the JEPA model, the latent head has nine fewer contact trials than its
direct head:−37.5 points, interval[−54.3,−16.7]. In contrast, JEPA-direct versus
supervised-direct has four more contact trials: +16.7 points, interval[−4.2,+45.8].
Thus the result is head- and controller-specific, not evidence that adding a
JEPA loss improves every readout. These are fixed contrasts on eight development
layouts, without multiplicity-adjusted confirmation. Three training seeds are
ensembled, not three independent physical replications per cell.

The named “rollout” head here uses **only its first .5-s latent transition**.
The controller does not search multi-step latent trajectories. This supports a
local predictive-head contribution in this setup, not the ultimate long-horizon
planning claim. The earlier negative [temporal comparison](go2_temporal_rgb_body_learning_comparison_development_v1_result_2026-09-05.md)
and [one-choice pilot](go2_online_choice_maze_pilot_development_v1_result_2026-09-05.md)
remain valid. They used different prediction/decision horizons, seed aggregation
or control loops; the new endpoint does not erase those results.

The JEPA latent head still makes four contact stops, fails the release-motion
criterion in two additional completed trials, and has three completed trials
with negative signed directional displacement. It requests stop only once in189
choices, so its lower contact count is not explained by always stopping. It
makes24 action changes and six moving-to-reverse transitions; retreat can avoid
contact while failing to make directional progress. The comparable direct-only
head makes60 changes and12 moving-to-reverse transitions. These are descriptive
mechanism observations, not a causal attribution of the contact difference.

## Evidence and integrity

The full audit covers1,114 actual-packet choices,10,671 RGB packets,63,307 sensor
samples and633,275 physics samples. It independently replays every model choice,
checks applied versus prospective commands and slew, reconstructs native
contacts and body histories, verifies camera/body geometry and paired prefixes,
and recomputes motion/contact masks and scalar endpoint errors.

The first full checker failed on a terminal contact that landed exactly on the
50th physics sample of a command. The producer correctly halted before its
post-command ingestion callback; the checker expected that callback to occur.
The [checker-only correction](go2_successive_choice_audit_clock_boundary_correction_2026-09-05.md)
was tested and re-audited the same evidence. The original FAIL, interim36 PASS,
all checker versions and all physical outcomes are preserved. No tolerance,
action, label, model or physical source changed to obtain the corrected audit.

Root: `.generated/go2_successive_choice_maze_development_v1_attempt_001`.

- Physical result SHA-256:
  `bd97ff363da74beebcb4a4770e9182b6554c791167a5a360d8535155d2bf8c7f`.
- Corrected full audit (`raw_artifact_audit_clock_boundary_v2.json`) SHA-256:
  `659c7d552ef49e0e9328a341da60fb97f2160639c33ba8fc7495783d4dba6d1f`.
- Full audit source/input dependency witness SHA-256:
  `f91c7537ad8897fdf835545a05fe557d73acc56c4fe568ad8da8c73767e28398`.
- Preserved failed predecessor audit SHA-256:
  `19c67f2b3fea7f614cacafcb1d480b021349eca655446b4de8f37952cb862fbd`.
- Physical launch SHA-256:
  `e9a4bd281e631f06e01a134c3d68b969613d4599bf0554299a55d34bc20f7bf5`.

Mean three-seed/five-candidate adapter times are22.45–22.59 ms for learned direct
heads and24.25–24.40 ms for latent heads; their95th percentiles are23.45–26.12 ms.
These exclude image acquisition and per-tick ingestion, and physics waits for
computation. They are not an end-to-end hardware latency guarantee.

## Next scientific intervention

The completed [action-coverage diagnostic](go2_successive_action_coverage_development_v1_result_2026-09-05.md)
over all610 training windows found80 initial stop-context
windows (16 per future action) and530 later windows, all repeating the previous
action. There are no later off-diagonal action pairs. All189 actual later action
switches lack that later training pair. Switched-group errors vary by method;
this is not proof that switching caused failures. Next collect matched moving-prefix
alternative suffixes before another training comparison. A pure specification
now enumerates384 new off-diagonal suffixes on the existing24 development
layouts, reusing96 observed moving continuations and120 initial branches. This
is600 planned context/action cells—not600 new independent environments or600
already observed outcomes. All96 proposed one-second source contexts exist.
The collector, prefix matching, raw audit and fixed protocol must be implemented
before launching that dataset; no new training or dataset run has started.

Continue the [follow-through plan](go2_successive_choice_followthrough_plan_2026-09-05.md):
qualify translation and maze-scale reorientation, obtain observed exits and
uncertain place associations, integrate beacon discovery/return with directed
executed memory, then perform independent final-maze and bounded real-platform
evaluation. The present direction cue is not a fixed point goal; teacher
initialization, ideal sensing and privileged native emergency stops remain
explicit experimental limitations. The ultimate scientific goal is not complete.
