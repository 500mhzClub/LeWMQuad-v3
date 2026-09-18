# Planning-map persistence comparison: actual-packet prefix result

Completed the bounded causal replay; no native scene or new command execution.
Session79952closed successfully after23.671889048069715seconds of replay/result
work. It consumed eleven actual paired observations, frames0–10, from the fully
audited eleventh maze0 episode. The first requested-command difference is10.
No observation or decision following that changed request was consumed.

At10 the original persistent-map controller requests left_turn[0,0,.45]; the
current-observation planning-map controller requests left_arc[.16,0,.45]. Both
remain active. All ten earlier requests match completed actual commands.
Raw/admitted visual state, accumulated map receipts, auxiliary partition,
mission/settling and executed residual histories match outside the explicitly
validated predecessor floor-reference wording. All eight compared raw six-action
forecast banks (frames3–10) match exactly. Public input arrays and the assigned
model weights are unchanged, with no parameter gradients.

The source of the changed decision is visible in its planning inputs:

| Recorded field at10 | Persistent planning map | Current paired observation |
| --- | ---: | ---: |
| Observed floor cells available to planning | 2036 | 1579 |
| Occupied cells available to planning | 90 | 75 |
| Reachable floor cells | 978 | 367 |
| Route cells | 35 | 31 |
| Waypoint X (m) | 0.47500000000000003 | 0.525 |
| Waypoint Y (m) | 0.025 | 0.025 |
| Left-arc utility (m) | 0.006524581440114514 | 0.006501887623268572 |
| Left-turn utility (m) | 0.006530222568216808 | 0.006470381162148236 |

Both proposals are OBSERVED_FLOOR_ROUTE_TO_FRONTIER and explicitly retain
unknown start-connector cells. The accumulated2036floor/90occupied cells still
exist unchanged in the candidate's retained evidence; only the planning view
uses1579/75current cells. Current floor-cell digest:
ea833ad917b25714a3101efedbbbc956d332e6fc1c67f0698ef5ffd787fb12fc;
current occupied-cell digest:
55597e20e4fea3552a077221356095da2bbd02e1d25e89d72c768ab2839b968d.

This establishes a causal decision difference under the declared planning-map
intervention. It does not establish which command would navigate better, a
memory advantage, a new arrival or any round trip. Tracking, contact history,
learned temporal state/residuals, mission/settling and scan state remain. The
small utility margins are recorded without converting them into physical gains.

Exclusive artifact root:go2_current_observation_planning_prefix_v1_attempt_001.
Result SHA-256:
8d0e1391f95c71cd71394356c902a9367a1d37fad75571197456dd0b0a6ca90a.
Launch SHA-256:
f2ccd1cc505c8b076f7006fdc96ee30e6fc2d997dfea97d05e911477aa039735.
Decision stream SHA-256:
d8a7e1b357e835a55a5855a1ac1ccc82b3c263d8cd98100e9d2a6a051666a735
(647020bytes). All1670source bindings and input artifacts verified before/after.
Model state remains
4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6.

New replay sources are now frozen by that launch:

- lewm/current_observation_planning_prefix_development.py:
  bfa297022c66fdf894ae02160af8f491d141710c2403f04089cf443b5e181549
- scripts/replay_go2_current_observation_planning_prefix_v1.py:
  4cc875eeb9b5838de9d1a3baebef64f71addc9c63e5a3bc071702edec76f7ca9
- lewm/tests/test_current_observation_planning_prefix_development.py:
  8fbd57731646baeddce67f03b1395ec68592b823cdf9768d1f1c5381a0d166be
- docs/go2_current_observation_planning_prefix_v1_2026-09-09.md:
  dd4f4b167cced77bb3f26b7725e8f226ce2a7fa873c54e267d127e551861548a

Tests58496closed15pass2.03s. Hardware19269closed with72.200GBavailableRAM,
95.332GBartifactfree,16physical/32logical CPUs/full affinity,3.5%CPUbusy and
bothGPUs0%busy. Preflight35257closedpass1670sources;71.629GBavailableRAM,
95.331GBartifactfree, no output created and no model/scene execution. One
CPU replay/numerical thread was then launched beside the existing native audit.

The current native19976worker2447506remains active99.0%CPU at95m43s,
CPU94m48s,RSS10,664,380KiB; its full audit/prefix is still pending after budget
terminal collection. Its launch was source/runtime identity evidence only.
No pending outcome from that attempt was used by this replay.

Next for this comparator: prepare a separate native collector, full raw audit
and exact prospective decision/physical/public prefix comparison through10
(1250physical samples). A completed current native baseline is required before
that execution. Keep the existing fixed independent learned/reactive cohorts
and their order unchanged. Goal active;zero verified round trips.
