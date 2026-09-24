# Action coverage after the completed successive-choice panel

The separately bound, no-fitting diagnostic completed over610 training windows
and all1,114 audited online choices. It reads actual applied-command histories,
not just requested action names. This is exploratory explanation of a completed
development panel, not an independent evaluation or causal switching experiment.

## Measured coverage gap

The80 initial training windows share teacher-stop contexts:16 per future action.
All530 later windows repeat their last applied action: stop112, forward103,
forward-left102, forward-right101, reverse112. There are **zero off-diagonal
later action pairs**. Every first-half-second contact label used in this count is
observed; no training windows were omitted.

Online,189 of970 later choices switch actions. All189 have a past/future pair
absent from later training. Excluding the always-stop baseline, that is189 of802
later learned-policy choices. Pair presence would still not establish full
scene/history support; conversely, the initial stop-to-action examples are real
training evidence but do not supply counterfactuals after moving histories.

## Errors on actually selected actions

The following means reduce within layout before averaging the eight layouts.
Only observed pre-contact motion is scored. “Repeat” and “switch” groups visit
different states and have different contact/censoring populations.

| Method/head | Switch choices | Switch contact-positive horizons | Repeat / switch position error, m | Repeat / switch Brier |
|---|---:|---:|---:|---:|
| Direct-only/direct | 60 | 4 | .0252 / .0271 | .0446 / .1661 |
| Supervised/direct | 55 | 4 | .0223 / .0296 | .0347 / .1806 |
| Supervised/latent | 23 | 4 | .0278 / .0424 | .0467 / .1138 |
| JEPA/direct | 27 | 4 | .0199 / .0254 | .0691 / .0532 |
| JEPA/latent | 24 | 0 | .0371 / .0699 | .0622 / .0114 |

Motion error is higher in the switched group for every learned method, but
contact Brier is not uniformly worse: both JEPA heads have lower switched-group
Brier. In particular, the JEPA latent head's four physical contact trials cannot
all be blamed on its24 immediate switching horizons, which contain no contact.
These patterns motivate a controlled data intervention; they do not establish
that absent switching examples caused the physical failures, that JEPA ignores
actions, or that its advantage is solely due to reversing.

The per-choice targets account for41 control-phase contact events. The audited
physical tapes contain four additional contacts during the forced zero release:
layout00/right supervised-direct, supervised-latent and JEPA-latent, and
layout01/left supervised-latent. These remain in the45-contact all-trial report.
A .5-s selected-action prediction does not cover an additional future release;
new stopping data and an execution contract covering final braking matter too.

## Next implementation

The prepared moving-prefix specification enumerates384 new off-diagonal suffixes
on the same24 development layouts. Ten ticks of one moving action precede30 ticks
of a different action, then five release ticks. Together with120 existing initial
branches and96 observed one-second moving continuations, this would provide600
context/action cells (400 train,200 development validation), not new independent
layout replication. A read-only check confirms all96 reference moving contexts
exist, with observed first-half-second contact labels.

Next implement physical prefix reproduction/matching and the actual collector,
freeze its protocol, collect the new alternatives and independently audit the
composite dataset. No new alternative outcome should be fabricated from a model
or labeled as observed before execution. Then compare the same three training
conditions with matched image exposure and budgets; preserve both the earlier
negative results and the new bounded positive JEPA latent-head result. Pursue
observed-place/exit memory and translation alongside the physical collection;
neither a better Brier score nor this dataset completes maze discovery/return.

Root: `.generated/go2_successive_action_coverage_development_v1_attempt_001`.
Result SHA-256:
`2d643ffb0a6c724c9c43b5bb593aa728f20df24b82edb8e9cbf9917e22951f8a`.
Launch SHA-256:
`68ed1af2335301b081f7f6c88a7393bb2df2919e682da12d9dde946f20e80300`.
The launch binds the diagnostic source and corrected full physical audit/input
evidence. No fitting, checkpoint selection or new physical execution occurred.
