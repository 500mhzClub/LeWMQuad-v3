# Causal residual final-goal prefix: exact implementation verified

Both fixed corrected models completed two exact fresh replays. Causal
observations/map receipts, raw forecasts and every original surface/nominal
check matched the executed-horizon predecessor. Every new selection exactly
equalled the explicit final-goal score correction. Both model states stayed
unchanged and no controller failure occurred.

JEPA's first command difference was tick 174: left_turn [0,0,0.45] replaced
hold [0,0,0]. The replay included 175 observations and stopped before executing
that changed command or reading its successor. Online correction values,
residual source ticks, availability ticks and sample counts matched the fixed
prequential diagnostic exactly at all 172 eligible forecast frames, ticks
3–174. The new final-goal score operated at ticks 171–174. No terminal change
occurred within the prefix.

Direct's entire 57-frame trajectory remained unchanged in both passes. Its
43 eligible correction receipts, ticks 3–45, matched the diagnostic exactly.
The exact final target never activated. Original waits 36–45 and the terminal
no-feasible-candidate outcome persisted. No direct-model recovery is claimed.

Ten state/scoring tests passed in 1.97 s, including prequential equivalence,
immutable original forecasts, clock/request failures, window expiration,
contact/surface/phase/later-path vetoes and original arrival/budget/terminal
behavior. The native collector/auditor scope test passed in 2.04 s; two causal
readout-prefix tests passed in 2.05 s. These checks grant no arrival from an
unexecuted command or lower retrospective prediction error.

Under `go2_causal_residual_final_goal_prefix_v1_attempt_001`:

- Launch: `ae3bc169a455b14c2c72e7f332463893254336475c9e008014003a5d675e17e9`
- JEPA decisions: `7c7a08e65bdfa8ae957b0d83fb5ea322bb10d6cffa61f7797ac0591c0a28cd51`
- Direct decisions: `23ed7bb7b307ab8b2130e1c38b52c6fd0aa775f0ad9e06c9d1787d498ccf076c`
- Result: `16a0fa63d7699f43d52b078c8255187d585b077764164b8579aa3e0133188727`

The source closure contains 1,364 paths. All native, readout, diagnostic,
checkpoint/correction and source bindings were reverified. Replay plus final
verification took 269.1673247090075 s on one CPU thread. Terminal available
RAM was 81,555,660,800 B and artifact free space 60,241,616,896 B.

A separately declared prospective native pair is now justified to test actual
arrival with the unchanged thresholds and budget. No native execution, fitting,
independent-maze, real-time, hardware or full-goal qualification occurred here.
