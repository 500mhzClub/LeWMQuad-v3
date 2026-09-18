# Planning from current observations: native maze0 pilot complete

The current-observation planning pilot completed collection, raw audit,
prospective physical-prefix comparison and final artifact verification. It
stopped at observation92 after both cameras lost measured pose support; its
decision tick remained91. There was no arrival or round trip. Strict physical
visibility also failed and remains part of this negative result.

Session70604 exited0. Root
`go2_current_observation_planning_maze_pilot_v1_attempt_001`; result
`dca1757aa1358ca48bf5d40240337aafb43391d0306b376309e1b3239da89780`;
launch `d8ea49c914abbc4f50f0ac0d973062cb98896015a28172eb937aefea2777c65e`.
1,678frozen sources,657artifact bindings,694.1448035400826s after launch.
Worker wall523.383851601975s,peakRSS2,787,270,656bytes. No model weight change.

Collection:103paired observations,102completed commands,5,850physics samples,
ten terminal zero ticks. Schedule terminalSENSOR_OR_MODEL_FAILURE; no physical
or acquisition stop. The original raw evidence records
`neither retained anchor nor previous frame supports current pose`, with no
qualified reference and insufficient incremental rigid-pose matches. Floor
registration and planning cannot proceed without that current visual pose.

Raw sensor/model/controller-command reconstruction and actual command audit
PASS. Native traversal remains in startcell[-1,0], no crossing or arrival.
Strict visibilityFAIL; hard measurement failures at18and84. Each has one
primary-camera stable-interior bad ray, errors1.0238051709210438mm and
1.035692302838065mm respectively, exceeding the unchanged1mm limit. No near
occlusion failure at those frames, and no auxiliary visibility failure. No
gate was relaxed or negative frame removed.

The fresh intervention exactly reproduced1,250physics samples and11paired
observations through10, all ten earlier actual commands, all11complete saved
prospective decisions and eight raw forecast banks. The original turn
[0,0,0.45] became left-arc[0.16,0,0.45] at10; the changed command completed.
No following physical outcomes were borrowed. Physical prefix hash
`d44b7057a9a2122311565d506dbf9a224d6fbcc64873af81bbb1f1cb7a2540b2`.

| Artifact | SHA-256 |
| --- | --- |
| Collection result | `63807665e6e513edea837e1c9d77204e5d20e579279fa102f93e6e756ecfa556` |
| Raw audit | `fb46f7ed8468f19ae089c2ffbaf12b2bf40c6462c27877939bdd7c7b1ad50835` |
| Worker terminal | `7d122e80384e562ea0b6283808d0f223516819fd64bcb94c0c7b034f9bd03cd1` |
| Native prefix comparison | `c8a73327baede5a55f768780c110613b9882e0e8b36f13526f75336fd31334b7` |

This changes accumulated planning-cell queries to the current paired-camera
view while retaining tracking/floor anchors, contact history, prediction and
residual history, mission/settling and scan state. It is not a memoryless
controller. Different planned commands produce different later camera views;
an early tracking failure in this one run does not establish a general memory
advantage. The paired readout was submitted separately as49332 against this
result and original learned readout
`a46e6051b125347804df4aab948e68a6466007952f7529b4f316ae39b114728c`.

There are now20completed/raw-audited native episodes overall, zero verified
round trips. The queued residual-feasibility maze2 trial was submitted after
this parent exited. The separately prepared tracking maze1 native trial follows.
No navigation reliability, JEPA advantage, real-time or hardware claim follows.
