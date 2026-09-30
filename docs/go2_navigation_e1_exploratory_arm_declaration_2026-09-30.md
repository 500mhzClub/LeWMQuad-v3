# E1 confirmation and exploratory-arm declaration, 30 September 2026

**E1 confirmed (Andrew, 30 September 2026), as pre-declared in the [E1 launch plan](go2_navigation_e1_launch_plan_2026-09-29.md), with one declared addition.** This document is committed **before the exploratory safety check starts and before any sealed-set access.**

## 1. Primary arm (unchanged)

- **Design:** C3-v2 and C4-v2 on the 60 sealed mazes × 1 episode × 3 training seeds, per the launch plan.
  - Running-time cap: 157 h.
  - Stop-and-report checkpoints at the pilot and at each seed boundary.
  - Storage checks at every seed boundary.
- **Prominent caveat in the E1 report.** C3-v2 under-predicts cruising travel: its median predicted/true 800-ms ratio is 0.33 on the held-out closed-loop states of the [C3-v3 round](go2_navigation_c3v3_round_report_2026-09-30.md). Its C3 results should be read in that light.

## 2. Exploratory arm (declared now)

- **What runs:** C3-v3 (readout `85ab19ec…`) and C4-v3 (`992c22fb…`), **one seed** (the recorded versions), on the same 60 sealed mazes and the same episodes (episode 0) as the primary arm.
- **Label: exploratory.** It was added after C3-v3 failed the pre-declared no-regression criteria N1 and N2 of its round.
  - It never replaces the primary result.
  - The E1 report keeps it separate throughout: its own section and tables, no pooling with primary seeds, and no substitution into the primary comparisons.
- **Safety gate.** It runs first, before any sealed-set access.
  - C1, C3-v3 and C4-v3 each run once on the 10 unused round mazes (`c3v3_sets_v1` layouts 22–31, role `safety_check`), on the frozen V4 harness.
  - **The arm runs only if C3-v3 and C4-v3 both have zero disallowed contacts and zero hard violations.**
  - Also reported: operating-margin violations, hold rates per leg, and trap counts by the frozen mechanism rules (trap 3 both ways).
  - C1 is a reference, not a gate.
- **Separate budget.**
  - Its running time is re-projected from the safety check's measured C3-v3 and C4-v3 mission times, plus a 20% margin, as its **own cap**, with its own stop-and-report.
  - Its jobs are named `E1X …` in the active-wall ledger.
  - When the arms interleave, shared intervals are split between them in proportion to each arm's owner-wall seconds in that interval.
- **Storage.** Before the arm starts, the combined storage projection for both arms must leave at least 15 GiB above the 12-GiB reserve, and this is rechecked at every boundary.
- **Order:**
  1. the exploratory safety check;
  2. the primary arm's pilot (seed-1 block);
  3. then both arms interleaved by maze, if throughput allows: the combined projection stays within the primary cap plus the exploratory cap, and storage holds;
  4. otherwise the primary arm runs to completion first, then the exploratory arm.

## 3. Custody prerequisite for any sealed-set access

This requirement comes from the repository custody rules (`AGENTS.md`).
- **The model-facing process must never open, print, parse, summarise, index or recursively search a `sealed_*` directory.** The E1 episodes live in one: `sets/sealed_test_v1` under the capability artifact root.
- **Final-test custody requires operating-system isolation and a fail-closed, reviewed, frozen, one-shot launcher.**
- **The capability pre-registration's sealed custody** (`sealed_custody`) records "no runtime or rendering" and "no E1 execution authority" for the sealed set.
- **So E1 execution waits on a custody arrangement for the sealed arms.** This covers:
  - who owns and can read the sealed root;
  - how the launcher is reviewed and frozen;
  - that per-mission outputs containing sealed geometry stay in a custodian-owned root, with only reader metrics exported.

  Its design is proposed separately for Andrew's approval. Nothing touches the sealed set until it is in place.
- **"V4" in `AGENTS.md`** refers to the legacy Go2 generalization V4 (`config/go2_generalization_v4`, invalidated 10 July 2026), not to this programme's `v4_completed_support` harness.
