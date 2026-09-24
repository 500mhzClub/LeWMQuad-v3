# Explicit 45-degree controller prefix result

The fixed corrected seed-2026091001 JEPA and direct models each admitted all
18 audited frames with the explicit 45-degree auxiliary packet and map.
Two fresh model/controller replays per method produced identical decisions;
model state stayed unchanged. Cross-calibration rejection and inherited
collision/controller scope checks passed (2 tests, 1.64 s); the preceding
packet/map/replay suite passed 17 tests.

All 17 recorded native commands and terminal states matched each method's
30-degree predecessor. The final frame-17 proposal was right_turn
`[0, 0, -0.45]` for both methods and differed from the predecessor. That proposal
was not executed in this capture. All 18 observations therefore belong to the
same executed prefix. The complete native prefix was checked exactly against
both predecessor cases. Forecast differences were zero in every compared
channel; maximum observed XY error was 0.0013957027851192636 m for both methods.
Native pose was used only by the posthoc observer check.

This establishes integration on a recorded prefix. It establishes no prospective
trajectory, goal arrival, independent-maze result, real-time operation or hardware
qualification. Next is the prospectively frozen two-case native 45-degree probe,
with the existing corrected models, planner, constraints and bounded reobservation.

Artifact root: `go2_auxiliary_downward45_controller_prefix_v1_attempt_001`.
The run bound 1,289 source files and took 49.224909205920994 seconds after launch.

| Artifact | SHA-256 |
| --- | --- |
| launch.json | e5b4267b4ebe6f106af2e546575eb26593481a5f2c94d3c5421297b6786cee1b |
| seed_2026091001_full_jepa_decisions.json | ea6696e3a57abeb13b0b6db794b0a51f26fd3e184e20522208dcc740f6ecf1b2 |
| seed_2026091001_full_direct_decisions.json | c06e869a33ace5e0ff3f5de7e7a1ae76e8c21c672b7af718be58028e3c112fcf |
| result.json | 0b45e01cf4b0f25a54f96dda8c799ef14b261e673787f4312fb41750d9c9151b |
