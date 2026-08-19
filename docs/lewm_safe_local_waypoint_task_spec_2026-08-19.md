# SAFE_LOCAL_WAYPOINT_TRAVERSAL

## Purpose and boundary

The first planner task tests local movement through an enclosed maze while the final beacon is hidden or absent from the planner input. An oracle route generator supplies a temporary local intent; this substitutes for the future memory/router and does not test beacon discovery, global routing, or closed-loop generalisation.

## Interface

At each decision, the planner receives recent RGB/context tokens, recent applied-control history, the frozen twelve-candidate action bank, and

\[
g_t=(\Delta x_g,\Delta y_g,\sin\Delta\theta_g,\cos\Delta\theta_g)
\]

in the current body frame. The waypoint is 1–4 graph edges ahead and approximately 1–3 m away, subject to controller limits. At least half of the evaluation segments should contain a turn or occlusion and should not be solvable by one straight primitive.

## Episode and labels

One decision block is executed, followed by observation and replanning. A waypoint is reached within a frozen metric radius (recommended 0.35 m, to be confirmed from controller/maze scale) before a fixed budget of 12 planning cycles. Collision, contact, fall, stuck, clearance and termination definitions are frozen from the runtime contract. Labels include body-frame displacement, heading change, waypoint distance/progress, completion time, safety at H1–H3, and path-level failure.

## Primary outcomes

Waypoint success is primary. Secondary outcomes are route/geodesic progress, collision/fall/stuck rates, safety vetoes, abstentions, planning cycles, path-length efficiency, and reverse-progress rate.

## Proposed pre-registered gates

The true-future planner must achieve ≥0.70 waypoint success, ≥0.65 pairwise safe-progress ordering, calibrated safety AUROC ≥0.75, and no family with zero success. A predicted two-step planner is promising relative to one-step only if success is +0.10 absolute or progress is +0.05 m/cycle, collision/fall is not worse by >0.02, abstention is ≤0.20, and the result holds in at least 3/4 held-out families. These are development thresholds, not claims of deployment readiness.

## Later beacon modes

Known beacon: memory localises its node and supplies the next local intent. Unknown beacon: a frontier/visitation novelty layer chooses an exploratory waypoint, then hands control to this same local planner. Once detected, switch from novelty to route-to-known-goal mode.
