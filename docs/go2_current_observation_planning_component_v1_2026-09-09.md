# Current paired-observation planning-map comparison component

This separately named controller removes accumulated floor and obstacle cells
from spatial planning queries. It retains the same world model, temporal inputs,
executed residual, visual tracker, measured floor registration/transport anchor,
mission, settling state, selector scan state and accumulated contact evidence.
It is an ablation of
spatial planning-map persistence, not a fully memoryless controller.

CurrentObservationPlanningMap runs the original paired mapping/classification
methods and returns their original receipt unchanged. During that observation
it captures the two actual measured floor-coverage cell sets and reconstructs
current obstacle cells from both public depth clouds using the original height
band, stride, grid and camera-specific arithmetic order. It never selects
current cells by accumulated first-witness frame, never adds unknown floor,
and never erases the retained map, return indices, partitions, patches or later
contact-resolution ledger. The observation still completes the original ledger
update, and any failure remains latched.

An explicit read-only planning view exposes the current cells, current admitted
pose and original contact-memory object. The unchanged selector chain receives
that view for waypoint/terminal-target decisions, scanning, nominal action/path
constraints and reentry checks. There is no fallback to accumulated planning
cells. Surface/contact checks retain their existing history. The view rejects
stale access and records current cell counts/digests and both camera witnesses.
All original model scoring, contact policy, mission budget and failure rules
remain; only which measured cells are available to planning has changed.

The retained map is still computed to preserve exact contact accounting and
provide a matched observation pipeline. This is not a memory-usage or runtime
optimization, and retained visual/mission/prediction/contact/scan histories must be
declared in results. This component grants no claim of a planning-memory benefit.

Do not alter the current run or fixed learned/reactive independent cohorts.
Before a prospective experiment with this component, freeze its complete source
and protocol, replay actual completed public observations causally, compare all
unchanged state and executed commands before the first changed request, then
stop before consuming an outcome following the changed request. A fresh native
trajectory and complete independent raw audit are required thereafter. This
component preparation creates no native runner, replay result or success claim.
