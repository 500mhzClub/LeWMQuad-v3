# Sustained recovery candidate and first changed command

The separate sustained-reorientation controller and helper are implemented.
The focused and original-helper regression suite passed 44 tests in 1.98 seconds,
session 50957, exit zero. Tests cover measured target termination, wrong-direction
motion, the eight-command cap, fresh phase/surface/path vetoes, special recoveries,
observation gaps, goal changes, invalid headings, stale scores/clocks, preservation
of every original forecast/score/constraint field, and actual selector dispatch.

The saved-input checker reconstructed every original helper selection through
observation 406 and compared the candidate on exactly 407 original observations.
All preceding requested commands matched. The first changed requested command
is at observation 406: original hold `[0,0,0]` becomes continued left turn
`[0,0,0.45]`. The original first turn at observation 405 remains unchanged.
The candidate's initial model-derived turn target is 0.277254376 rad; measured
progress at observation 406 is 0.023910180 rad. The new command remains subject
to that observation's original full nominal-path and sampled-surface vetoes.

The checker verified a 1,967-path source closure and completed in session 43273,
exit zero. Its result is
`docs/go2_sustained_hold_reorientation_saved_prefix_2026-09-11.json`, SHA-256
`a58a89afd262ffaef63c5adac6a7ae6f8259f5b93a21f19a5dc4da4e4e9f6177`.
The preparation record is
`docs/go2_sustained_hold_reorientation_preparation_2026-09-11.json`.

This is a saved-input helper comparison. It has not rerun the complete controller
from raw sensors, executed the changed command, or consumed a following
observation as a candidate outcome. The next step is a prospective raw
complete-controller replay through observation 406 using the original assigned
model and this frozen candidate. A subsequent native test needs separate
resource and queue admission. The existing contact, tracking and extended-budget
experiments and independent-study definition remain frozen.

The candidate addresses the independently verified physical turn/hold reversal.
It does not by itself resolve the late floor-correction failure, establish that
any translation will become admissible, or qualify navigation, timing or hardware.
