# Partial height candidate and fresh prefix preparation

Goal remains unachieved: 22 completed/raw-audited native episodes, zero verified
round trips. The preceding status turn verified both live CPU jobs; it was a
verified wait, not another completed navigation experiment.

Implemented the typed scalar-height observation and explicitly rebound all three
measured-pose consumers, retaining the original registration, tracker and planner
code. Partial observations cannot change the full anchor, normal or rotation;
all current points retain the original 3 mm coherence gate and total pose
correction limits. Full-plane reacquisition remains original. No running or
previously frozen source was changed.

The earlier component-test output was lost during context compaction. Process
inspection confirmed no pytest remained live; a fresh focused verification
97394 then completed, exit 0: 24 tests passed in 10.20 s. No prior pass was
assumed. The new prefix/runner tests 64762 completed, exit 0: 23 tests passed in
3.80 s. These exercise unrelated earlier-state changes, exact raw tracker,
original boundary identity, live typed pose, retained anchor, changed request,
negative boundary, no subsequent packet, input/model mutation, truncated replay
and persisted comparison failures. Component tests include repeated partial
observations followed by original full-plane reacquisition.

Actual completed-native admission also passes (86761 exit 0), using the original
diagnostic admission function on the completed result, launch and raw audit.
This is not a substitute for the full launch-time source/artifact verification.

Fresh hardware in the same command: 16 physical / 32 logical CPUs, all 32 in
affinity, CPU busy 6.6%; 75,821,559,808 RAM bytes available; artifact free
75,850,260,480 bytes; workspace free 21,358,764,032 bytes. Both GPUs idle; card 1
VRAM total 34,208,743,424, used 1,398,722,560 bytes. Existing hold replay PID
2509963 RSS 3,742,539,776 bytes and support benchmark PID 2511382 RSS
4,295,270,400 bytes. Their 8+16 GiB allowances plus this 8 GiB CPU replay fit
available RAM. No native scene runs. The fixed supervised cohort's storage
gate remains unmet, exact cache cleanup approval remains pending, no deletion.

Frozen prospective sources for the exclusive first attempt:

| Path | SHA-256 |
| --- | --- |
| lewm/partial_floor_height_development.py | cef9ad005c7169a6f984494f21c85a2cdb5f38fa3905e5f6848dfc74fabc2fd5 |
| lewm/partial_floor_height_controller_development.py | 5b79f95e7b1aeadc62571900cea6d067efb686e0860dc2e73484180de4c00e51 |
| lewm/partial_floor_height_prefix_development.py | 3bbcfa7d52a6cabcf1783af00a3136c61744545531f24bdf3270116f610aebcf |
| scripts/replay_go2_partial_floor_height_prefix_v1.py | de8b8727c0f53c5239ad4d73b33a54b99227e67731f06e3f241bc576e2d483b1 |
| lewm/tests/test_partial_floor_height_development.py | 4e236e8f44009800d606b5b66614720454e3c4c0580770b0b54364c201e3ad02 |
| lewm/tests/test_partial_floor_height_prefix_development.py | e368b6aea9c6ed2265d74f4e9e3d7be6316c3c35d2672d49337ad59f2ec4c445 |
| docs/go2_partial_floor_height_prefix_v1_2026-09-09.md | a008a25168aded18f41ded99bd9f38b25e52ff7c76686afc4a487d4c938ba9b2 |

Command uses the existing Genesis environment, deterministic hash seed,
PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=.:lewm_genesis:lewm_worlds and one
OMP/MKL/OpenBLAS thread:

```
.generated/venvs/genesis_rocm_0_4_6_v1/bin/python scripts/replay_go2_partial_floor_height_prefix_v1.py
```

Output is go2_partial_floor_height_prefix_v1_attempt_001. It starts from the
original initial state and stops at the first original failure, frame 504,
regardless of whether the candidate succeeds. Submission does not establish a
changed command, physical safety or navigation. A future native experiment
requires its own prospective protocol and full raw/physical prefix comparison.
