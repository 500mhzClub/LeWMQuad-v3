# C0 executed-prefix fidelity check: erratum, 28 September 2026

Approved by Andrew Knowles in the [28 September handoff](go2_navigation_capability_handoff_2026-09-28.md) §3, with his rulings of the same day. Recorded before any further run. Machine-readable form: [JSON](go2_navigation_capability_oracle_prefix_erratum_2026-09-28.json).

## Why

The check exists to catch restoration errors in C0's physics branches. The original requirement was to compare "every actually executed candidate prefix" against its branch under the matching command tape (protocol `oracle_check`).

The frozen checker (`lewm/navigation_capability_oracle_development.py:verify_executed`) also failed any branch whose matching prefix was zero ticks long. When the dispatch layer substitutes a different command on the first tick of a decision, every branch shares the unexecuted committed prefix. That is a vetoed prediction, not an executed one. It says nothing about restoration fidelity.

That happened once in C0 02/0. At the 119.9-s source boundary the current observation was missing on the first tick, and vetoes followed. All six branch rows at that one decision had no match, while all 2,322 comparable prefixes matched exactly.

## Rule

For each decision (branch source boundary):

1. **Comparable branches.** A branch is comparable when its applied command tape equals the actually dispatched commands for at least one 20-ms tick from the source boundary. It is compared over that matching prefix only. This is usually the selected candidate, but may be another, such as hold.
2. **Tolerances are unchanged:** 1 mm position and 0.1° yaw over every native 2-ms sample of the matching prefix. **Any mismatch on a comparable prefix remains a stop for C0 validity.**
3. **Non-comparable branches.** A branch with a zero-length matching prefix, at a decision where another branch is comparable, was simply not executed. It is recorded, not judged.
4. **No matching branch.** A decision where no branch is comparable is counted per episode with its cause, and neither qualifies nor disqualifies the episode. The cause is the dispatch-layer substitution on the first tick:
   - **missing current observation:** `CURRENT_OBSERVATION_UNAVAILABLE_OR_STALE`;
   - **veto:** `COMMAND_WINDOW_VETO_LATCHED`, `CURRENT_OBSERVED_OBSTACLE_VETO` or `COMMITTED_PREFIX_NOT_EXECUTED`;
   - **override:** `FLOOR_REACQUISITION_HOLD`, `NO_ON_TIME_PLAN` or `MISSION_SETTLING_OR_TERMINAL`.

   The substitution sequence over the following ticks is also recorded. If the first tick is ordinary plan dispatch (`CURRENT_NOMINAL_OBSTACLE_TEST_PASSED`) or has no request record, the absence of a match is unexplained, and that is a stop for C0 validity.
5. **Completeness.** Every branch receipt in `model_calls.json` must have exactly six comparison rows, and vice versa. A zero-length row must also be confirmed independently from the retained first-tick branch command against the dispatched command. Any inconsistency is a stop.
6. **Veto reporting.** Each C0 episode reports its dispatch-substitution tick counts by reason. It also counts vetoed selections: decisions whose served window contained a veto or missing-observation substitution. Repeated vetoed selections in a failed episode are diagnosed as a harness mechanism.

## Scope and limits

- **What changes.** This changes the evaluator-side qualification of C0 records only. The harness, controllers, C0 oracle, frozen checker, thresholds and all recorded outputs are unchanged.
- **How it is applied.** A new evaluator applies the rule to the frozen checker's preserved rows, `requests.json` and `model_calls.json`. It runs for every C0 gate and validation episode, including the four already qualified.
- **02/0.** Episode 02/0 reached a terminal mission outcome, so it is re-evaluated from its preserved records under this rule and is not rerun. Its navigation outcome stands as the unchanged physical reader finds it.
- **What stays out of scope.** C0's branches still do not model the dispatch veto. Changing that would change the oracle's definition and needs Andrew's approval.
