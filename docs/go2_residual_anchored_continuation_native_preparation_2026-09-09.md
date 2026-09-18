# Anchored continuation native sources frozen; preflight passed

The completed prospective replay changes hold to left arc at observation180;
its exact result and independent verification are recorded in
docs/go2_residual_anchored_continuation_prefix_result_2026-09-09.md.

Separate native source files:

- scripts/residual_anchored_continuation_maze_episode_development.py
- scripts/residual_anchored_continuation_maze_audit_development.py
- scripts/residual_anchored_continuation_native_prefix_development.py

Collector and audit derive from the original residual maze2 implementations,
with the explicitly different anchored controller and result labels/flags. The
prefix admission reconstructs all181 saved decisions against the original
completed residual native episode. Physical comparison is bounded to9,750
samples,181 public observations,180 preceding commands and178 forecast banks.
It compares no physical outcomes after changed command180.

The separately named launcher and protocol are now complete. Focused boundary,
mutation and original collector/audit calculation tests pass25 cases. Launcher
admission and fresh-model worker/failure-retention tests pass29 cases, including
the final worker change retaining the written prefix binding before subsequent
verification. Combined54-case run76990 passed; after that last retention change,
launcher-only86888 passed29 cases in2.70s. Earlier42614 passed29 cases.

Frozen new source identities:

| Path | SHA-256 |
| --- | --- |
| scripts/residual_anchored_continuation_maze_episode_development.py | 4e98a79ea5ad16c1d2021deae52abd2aec0a38d4ace1df80d2ca3994d990e949 |
| scripts/residual_anchored_continuation_maze_audit_development.py | 6ae658b91a0859b8d4b7ae99ba9da22dc2541183c4a52409301bd857a24675f3 |
| scripts/residual_anchored_continuation_native_prefix_development.py | 673b4f62e601c2270b17dcac5ce07888e75bbeb51f81531fe74d38c69bfcf30c |
| scripts/run_go2_residual_anchored_continuation_maze_pilot_v1.py | c9db3b3c1e5fc36afdff7dd1ed82e83756cbad35c450aca347942e9be48dad4a |
| lewm/tests/test_residual_anchored_continuation_native_development.py | 83ee1f42fefb8a396c1610efab3aff6b15b15f3ad6e731af576ae057b3647f88 |
| lewm/tests/test_residual_anchored_continuation_native_launcher_development.py | 9e189206e21bc82ffa4dc1275b67d37fd7b82752bd85b8546e95eba020a32391 |
| docs/go2_residual_anchored_continuation_maze_pilot_v1_2026-09-09.md | 3a9c9c52bb1c2c7cf58698e16598b9a77008ea4c25b70d2feaa7e38016bd989c |

Hardware73459:16 physical/32 logical CPUs with full affinity,3.5% busy,
51,326,849,024 bytes available RAM; artifact volume709,833,240,576 bytes free,
workspace21,358,272,512 bytes free. card0 GPU0%, card1 GPU8%, card1 VRAM
1,853,222,912/34,208,743,424 bytes. Supervised parent2534319 and its single
spawned worker2535576 are live. One8GiB read-only preflight may overlap that
worker's conservative32GiB allowance; no second native scene will launch.

Next authenticate the full actual prefix/prior/source/environment chain in
read-only preflight. No anchored native scene or navigation outcome is claimed.
The queue remains supervised layouts1,2,3, then prepared direct-flow maze3,
then this anchored maze2 development experiment. Preserve the running supervised
process19047, completed replay, frozen sources and every original failure.

Preflight35981/PID2537733 exited0. All1746 source bindings and original input
conditions passed. The fresh scope handled1,770,739 digest requests across
137,894 unique files; initial and final hashing each covered64,656,470,245 bytes,
with1,632,845 guarded reuses and25 isolated verifier functions. Every cached
path was freshly rehashed; no retained cache or imported-global mutation.
No output or scene was created. Both original native resource admissions pass:
52,422,590,464 bytes RAM and708,495,810,560 artifact-volume bytes available.
Workspace free21,358,264,320 bytes; CPU3.5%,16 physical/32 logical with full
affinity; card0 GPU0%, card1 GPU7%. Supervised parent2534319 and single worker
2535576 remained live. This pilot is ready for its recorded queue position;
recheck live ownership and resources before actual native launch.
