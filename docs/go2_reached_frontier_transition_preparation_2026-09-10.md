# Reached-frontier transition preparation

Implemented a separate observed-grid frontier transition following the completed
maze3 diagnosis. The original running and completed controller sources are
unchanged. This component has not been run in a native scene or full raw prefix.

`reached_frontier_waypoint_development.propose` retains the original connector,
inflation, reachability and path reconstruction. Its only target-selection
change is an explicit visited-frontier exclusion. Excluded frontier cells remain
usable as path cells; an observed mission-goal cell is never excluded.

`ReachedFrontierState` marks a selected frontier visited when the current
registered body position enters that observed floor grid cell. This uses the
existing 50mm grid partition, not a tuned new metric arrival radius. It preserves
the original proposal until a visit or retained exclusion changes the target.
A visited target becomes eligible for reconsideration when the observed
floor/occupied classification of that cell or a cardinal neighbor changes.
Unchanged distant geometry does not reset its visit. A mission-goal change
starts a new frontier search while retaining the actual map/contact state.

If all reachable frontier targets have been visited, the returned route is
empty and the existing bounded view-scan policy applies. Retirement is not
obstacle evidence, a coverage certificate or proof that exploration is complete.
The same-frame operation is idempotent, rejects changed input within a frame,
and records visits/releases with the observed clock. The ledger is bounded by
the original 40,000-cell planner input bound.

`ReachedFrontierRecentQualifiedController` changes only the mapper's waypoint
proposal behavior over the completed original RecentQualifiedDirectFlowController.
It keeps the existing observation, floor registration, contact history, mission,
selector, model, action bank and clearance constraints. Its additional receipt
records the last observed frontier transition explicitly.

Tests: handle 35600, exit 0, 15 passed in 4.23s. They exercise original proposal
equivalence before retirement, retained route traversal, unchanged start and
connector clearance, local-evidence reconsideration, mission-goal handling,
all-visited view transition, same-frame guards, and exact original warmup
observation/registration/contact/mission behavior. No scene or neural inference
was used by those tests.

Before starting its full-controller prefix replay, the original six-model
native workers exposed a separate shared startup failure: the planner's old
wrapper-class guard rejects the admitted expanded correction wrapper. All six
stopped at observation3 before model prediction. That interface failure is now
the immediate priority; its adapter and four-observation raw replays are
separately named and do not include this frontier intervention.

Next for this component: authenticate and replay the completed original maze3
with identical models/public observations, preserve complete original decisions
until the first requested-command or terminal difference, and verify unchanged
observed/contact/model state and raw forecast banks. Stop at that boundary,
then require a separately reviewed fresh physical continuation before any
navigation claim. Keep these development corrections separate from the still
unexecuted independent-layout population.
