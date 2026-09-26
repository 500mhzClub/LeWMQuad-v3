# Corrected C1 screen in progress

The one-time settled-start task correction is frozen in `13ebc8e2`. Its all-180-episode, both-target initialization test passes at 1 nm. The original five pilots remain pre-fix records; their corrected-reader verdicts did not change. The current source/controller owner remains unchanged throughout this screen.

| Episode | Round trip | Contacts / hard violations | Finding |
|---|---|---|---|
| 00/0 | Pass | 0 / 0 | 130.42 simulated seconds; 273.03 source wall seconds; 59.20 MiB of retained logs/hashes; no sensor frames or snapshots |
| 01/0 | Fail | 0 / 0 | Mapper rejected the first observation: initial measured floor unavailable |
| 02/0 | Fail | 0 / 0 | Unchanged mission constructor rejected the target outside its ±4.9-m initial-frame map bound |
| 03/0 | Pass | 0 / 0 | 463.12 simulated seconds; 924.82 source wall seconds; 694 motion-dependent footprint/view holds out of 715 total holds / 1,148 decisions |
| 04/0–09/0 | Pending | Pending | Fixed original assignments continue serially |

Episode 03/0's minimum articulated separation lower bound was 112.85 mm; every sampled state and secondary interval check passed both thresholds. The holds are logged mechanism descriptions, not counterfactual evidence that each excluded candidate was physically safe.

The map-bound rejection predates the reference correction: episode 02/0's nominal target coordinate already reached −5.2903 m; the corrected cue reached −5.2805 m. The registered world geometry and target-reference equality remain valid. This is a deployed-harness capability limitation, not an episode regenerated to remove a failure.

The initial batch wrapper stopped too broadly on controller failures. Accounting continuations `c3476693` and `75cfd038` preserve the original source owner, assignments, physical traces and outcomes and run only previously unrun assignments. Neither failed mission was retried. Initial perception failure consumed one frame even though the original completed-acquisition counter was still zero; retained packet hashes and pose/mission rows preserve that evidence. The constructor rejection occurred before sensor consumption and is also counted as an assigned mission failure.

The active process is `scripts/continue_go2_capability_correctness_screen_development.py --stage screen`. Runtime receipts are under `cohorts/v0_task_c1_C1_screen` in the programme RecoveryStorage root. It completes the remaining assignments, evaluates each natively, writes the complete result and re-projects the budget. The screen can no longer reach 9/10. After completion, classify all failures/stalls, choose the dominant mechanism, make one allowed shared harness change, freeze the next version, and rerun the prescribed development screen. Do not start C0 or validation on this failed version.

Prepared while the screen runs: an official-video renderer using seed/command regeneration and every consumed packet hash, keeping regenerated ego frames in memory; and a read-only oracle coverage helper reproducing the existing 5,460-admitted-interval pilot receipt exactly. The renderer has only been syntax-checked; neither helper has launched new physics or a new science episode. They need binding to the eventual passing harness before use. These preparations do not constitute official videos or a passed oracle gate.
