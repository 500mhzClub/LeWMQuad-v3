# Longer-budget prospective comparison semantics tested

The prepared comparator requires the original 4,000-step chained single-pass
controller to reproduce every recorded chained native decision using only the
existing performance normalization. The extended controller must then match
that single-pass decision after changing exactly its controller declaration,
new implementation flag and two declared budget scalars. This module performs
no input admission, actual recorded-history replay or native execution.

## Source and test evidence

| File | SHA-256 |
| --- | --- |
| `scripts/extended_return_budget_prefix_comparison_development.py` | `250fb09c30214e95a0d681ffaa7aeff755d168056e9f285995b702c4f4cc3ea9` |
| `lewm/tests/test_extended_return_budget_prefix_comparison_development.py` | `be27f346d7ab3394088d70e18758ce1b47669b6e4eb6ebced383c6b5c20d0a90` |

The focused suite passed **17 tests in 54.63 seconds**, tool session 42023,
using the original deterministic single-thread environment and
`pytest -q -p no:cacheprovider`. The initial invocation, session 30722,
stopped during test collection with **one import error in 2.22 seconds**:
the test omitted `_v1` from the existing ground-plane helper module name.
That import was corrected without changing the comparator.

All 2,639 sources bound by the live chained/single-pass comparison were
independently rehashed afterward and matched. The two new files are outside
that roster. The tests overlapped the already declared non-isolated CPU
comparison; they establish no isolated performance result.

## What was checked

A four-observation actual synthetic image/depth sequence drives two independent
fixed-head test models through the real chained single-pass and extended
controllers, with respective mission budgets 4,000 and 8,000. Every complete
decision and normalized retained state agrees, both models make one forward
call, weights remain unchanged without gradients, and articulated geometry
agrees. This covers different-budget behavior in the short sequence, not the
complete recorded native history or late raster history.

The retained-state serializer preserves every controller field except the
model and geometry, which a replay caller must authenticate separately. It
changes only ten explicitly located implementation type tags (seven distinct
classes, with memory/residual aliases represented in multiple locations) and
three mission-budget scalar locations. It validates the actual shared memory
and residual aliases. Motion fields, mission progress, history, unknown fields
and identical-looking budget keys at other locations remain visible.

A separate synthetic comparison population advances both actual mission
objects through all 4,004 observations up to the old deadline. It uses a
warmup decision template and supplied positions, so it is not a full
sensor/controller replay. The first normalized difference is the old mission's
terminal decision at frame 4003. Both requests in this fixture remain zero;
the candidate remains nonterminal. The comparator stops there and rejects a
subsequent observation. It never interprets this as physical progress or a
verified round trip.

Any earlier complete decision difference also stops comparison and prevents
a claim of budget-only preboundary behavior. Tests change commands, terminal
state, unknown tracking evidence and unrelated nested budget fields. Changed
recorded decisions, frame identities, integer budgets, controller declarations,
model-call counts and invented forecasts are rejected. Changed motion/mission
state and broken aliases remain detectable.

## Required integration

The full recorded-prefix runner and its closed-output checker still need to
bind these comparison rules to authenticated public packets, actual model
hooks, separate model/geometry identities, physical command endpoints and
fixed retained-state checkpoints. They must stop reading immediately after
the first normalized decision difference or terminal boundary. The current
module explicitly reports that it has not performed those checks.

Dispatch must follow completion and authentication of the already running
4,014-observation chained/single-pass comparison. No new CPU replay has been
launched, queued or admitted by this preparation. A fresh longer native trial
still requires the resulting prospective evidence, resource admission and a
bound protocol/launcher, followed by complete physical-prefix and outcome
audits. No old terminal episode is resumed.
