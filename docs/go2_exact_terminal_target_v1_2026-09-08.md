# Exact terminal mission-target diagnostic V1

Authenticate the completed overlap-retention direct-039 native result and its
full raw readout. Process all 46 selection contexts among its 240 recorded
observations. Use only current observed pose, map-frame receipt, the fixed
initial-body [1.2, 0] m mission goal and already audited candidate predictions
and constraints. No model inference or native pose is needed for scoring.

Only when the proposal is OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL and the selected
waypoint exactly equals its final cell centre, substitute the exact transformed
mission point. Require that point to remain in the same observed goal cell.
Recompute the unchanged half-second distance/bearing/contact utility and select
among the same six candidates with the same surface and nominal vetoes. Preserve
all predictions, constraint records, observed route and arrival criteria exactly.
Intermediate and frontier targets remain unchanged, including a goal cell later
excluded by conservative grid inflation. No direct shortcut is introduced.

Save all contexts, original/conditional actions, target offsets and exact revised
selection records. These are conditional decisions on an executed old history,
not a changed trajectory, arrival result or navigation claim. Five focused tests
cover target ranking, unchanged inputs, both veto types, frontier/intermediate
preservation, out-of-cell rejection and empty feasibility. No threshold search.

Bind this protocol, helper, checker, tests and preceding native result report;
verify source and input identities before and after. Use an exclusive new output
root and preserve terminal failures. Require 4 GiB RAM, 40 GiB storage reserve
and a 256-MiB output allowance. Record hardware before and after. The 46 small
contexts run sequentially; there is no independent native or model workload to
parallelize. No training, new commands, physics, retry, old-result changes,
real-time, hardware, support, uncertainty or navigation qualification.
