# Robot-visible primary prefix compatibility result

Both fixed corrected seed-2026091001 full-JEPA and full-direct models reproduced
all 19 executed commands and the unexecuted final proposal on the new primary
RGB/depth/body prefix. All 20 observer/controller frames were admitted. Two
independent fresh model/controller replays per assignment produced exactly
identical new records, with complete model/correction state unchanged.

The primary depth and physical/public histories remain exact; primary RGB
differs from the robot-hidden predecessor. With the changed RGB, maximum
observed XY error over these 20 frames was 0.0013957027851192636 m, versus
0.0021068214643736064 m for the predecessor. Maximum absolute XY forecast
differences were 0.0000010132789611816406 m for JEPA and
0.00000011920928955078125 m for direct. These small changes did not change any
recorded-prefix command. They are observations on one short reused prefix,
not a general robustness or model-quality claim.

Auxiliary depth did not enter this compatibility check. No model was trained,
no new native command executed, and no complete mission was evaluated. The
twentieth proposal was not executed in the sensor capture and is reported
separately from the 19 matched actual commands.

Artifact root: `go2_visible_robot_primary_compatibility_v1_attempt_001` under the
fixed navigation development artifact root.

- Launch: `7610b502ccb96ea00c97c5400e8128bc7a034c71605a4c8f3aa5fe721b388967`
- JEPA decisions: `0ae0ea66b3adb17dd35fd373925ac4409f319747e7ea2d98c6ab0c2d25ee4212`
- Direct decisions: `e705d38314a4863426cdd1348c767422b8577c1d91c0280b92c71e31b834f77a`
- Result: `7b524b6b1408db532662f2fd7a190ed7a47c726f7636daa40e3651eff188d186`

All source/input/correction/model bindings reverified. Replay and analysis took
37.94922790792771 s after admission. Two focused causal-boundary tests passed in
0.11 s. This supports a separate integration test of the measured auxiliary
stream with unchanged models and observer. It establishes no full-mission,
independent-maze, latency, hardware or overall-goal completion.
