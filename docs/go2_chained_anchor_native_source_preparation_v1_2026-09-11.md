# Chained-anchor maze2 collector and auditor: source preparation

Prepared separate `chained_anchor_maze02_episode_development.py` and
`chained_anchor_maze02_audit_development.py` from the exact completed original
no-RGB JEPA direct-flow sources. Changes are restricted to the controller class,
declared collection status and the explicit chained-anchor capability flag.
Both use the same `ChainedAnchorResidualController` currently undergoing full
controller prefix replay.

The complete original physics loop, 3000-tick navigation budget, warmup and
terminal drain, sensor contracts, renderer witnesses, storage guards, gains,
friction checks, artifact persistence and timing records remain unchanged.
The raw audit still replays every complete controller decision from the recorded
public sensors and original model, verifies actual commands and contacts, and
requires the original independent traversal/arrival and strict visibility gates
for any round-trip result. Native pose remains evaluator-only.

Focused tests compare the entire abstract syntax trees with the SHA-bound
originals, allowing only those declared differences. Mutation checks ensure
that shortened execution, substituted commands, missing raw decision comparison,
changed model-state checks and relaxed visibility/success gates are rejected.

No simulation was launched or queued by this source preparation. Before any
prospective run, the full-controller result must complete and be checked; a
physical-prefix comparator and source-bound launcher must then be prepared.
The existing diagnostic simulation queue must finish in its established order
through the budget, sustained-turn and contact-plus-flow experiments. A future
chained-anchor experiment must preserve those outcomes and the original failed
tracking attempt. It does not automatically select the policy for the
independent-layout study.
