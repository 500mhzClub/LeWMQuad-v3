# Residual hold-veto readout preparation

A read-only inspection of the completed residual maze 2 decision stream
(6528 exit 0) authenticated the exact native result, decision stream, command
tape, raw audit and current source bindings. All 3,014 decisions were ordered
and matched actual completed requests. Of 2,643 selected holds, 24 had no
strictly better allowed nonhold; 2,619 had a better-scored alternative rejected
by an original geometry veto. No better originally feasible alternative was
found. First examples occurred at frames 32 and 180 respectively. These initial
counts read saved fields; the new helper additionally reconstructs the scores.

The original first-point feasibility correction changes predicted point 1 only.
In the original eight-segment planner this can change segments 0 and 1, while
segments 2–7 retain both endpoints. The new readout reconstructs every saved
hold utility and component from raw predictions and causal residuals, verifies
all ordered original veto receipts, and distinguishes immutable later-segment
or preserved surface vetoes from first-two-segment-only vetoes. It does not
evaluate a new corrected trajectory or infer unexecuted physical outcomes.

22 focused tests passed in 2.04 s, session 24089 exit 0. Actual original
constrain/plan functions demonstrate unchanged later segment receipts after
the first-point correction. Tests also cover utility versus geometry causes,
future/altered residuals, forged scores and components, malformed forecasts,
path ordering/radius/summary mismatches and complete command populations.
The runner's exact terminal label was checked against the completed native
readout: MISSION_TICK_BUDGET_EXHAUSTED at frame 3003.

Fresh hardware 23500 exit 0: RAM 72,535,187,456 bytes available; artifact free
74,959,761,408 bytes; workspace free 21,358,551,040 bytes; CPU 6.6%, 16 physical
/32 logical/all affinity. Card 0 GPU busy 0%, VRAM 414,113,792 bytes; card 1 busy
1%, VRAM 1,398,722,560 of 34,208,743,424 bytes. Native parent 2518502 and its
single worker 2519242 are active; worker RSS 2,948,550,656 bytes at the hardware
snapshot. Hold V2 RSS 7,145,590,784 bytes. Native32 + hold-growth16 + readout8
GiB allowances fit measured available RAM. No second native scene is created.

Frozen files:

| Path | SHA-256 |
| --- | --- |
| lewm/residual_hold_veto_readout_development.py | a59fc3811dd9a6470a0fb2e49c20ea2dc897e57c7405ab0c46f1dd007c26a066 |
| lewm/tests/test_residual_hold_veto_readout_development.py | fc777d9756a751306fc7564ffc83ddc413be85fec80908b87505fb37c3ee7a04 |
| scripts/read_go2_residual_hold_vetoes_v1.py | b057de4e10b72dbd76332c341b588d4044c47dae8c39038e713fdcae5669d7cf |
| docs/go2_residual_hold_veto_readout_v1_2026-09-09.md | 8d4725dc007c22ff76fa74fa96f406ec78887c77700a3de81e05606d072c7e17 |

Run scripts/read_go2_residual_hold_vetoes_v1.py in the existing Genesis
interpreter with deterministic hash seed, disabled bytecode writes and one
OMP/MKL/OpenBLAS thread. Exclusive output:
go2_residual_hold_veto_readout_v1_attempt_001. The completed scoped-verifier
benchmark is admitted before using fresh per-call digest scopes. Every
original native verifier condition and final fresh digest check remains.

The live hold V2 replay is not an input and has not been called a completed
negative result. The current native height experiment is separate and unchanged.
No new policy, model training, deletion or future physical outcome is included.
