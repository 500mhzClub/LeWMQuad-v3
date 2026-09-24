# Ready-for-approval storage consolidation

Verified saving:7,219,273,728 allocated bytes (6.72 GiB), from27,210 duplicate
paths in5,523 groups. Every original filename and byte remains available by
sharing identical storage through hard links. All affected files are preserved
scientific artifacts and must remain immutable; editing one linked file in
place would also change the other names sharing its inode.

Exact byte-verified inventory result:
732b0a95a98d598843e1d56f85612093b3d78e3e1ed145f4e27d24f2ceb6d577.
Exact proposal.json containing every canonical/duplicate path and metadata:
0a0b273b47b4d0df3b0ce76104b0cb348574e339b6dfa0ee235ab3e15a890bb7,
in go2_training_artifact_duplicate_inventory_v1_attempt_001 under the established
navigation development artifact volume.32,733 paths were freshly hashed against
their original artifact bindings. Independent1482 rechecked inventory identities.

Consolidation preflight86724 exited0 without creating output or changing inputs.
It rechecked all reviewed hashes/metadata and confirmed no other known development
worker or accessible artifact descriptor. Four identified unrelated services
deny descriptor inspection: systemd3119, sd-pam3120, fusermount34055 and
ssh-agent2176426. This limitation is recorded; universal process quiescence is
not claimed. Native and replay jobs are completed, and none may start during
consolidation. Synthetic operation/failure/process checks:14 tests pass83383.
The earlier stricter preflight27043 failed on protected systemd descriptors and
made no changes; the revised process-check scope is explicit in the protocol.

Free artifact capacity at final preflight:72,989,745,152 bytes (about67.98 GiB).
Adding the verified saving gives about74.70 GiB, above the frozen supervised
three-maze requirement of73 GiB. Actual recovered space and the unchanged
resource gate will be rechecked after conversion; unrelated usage can vary.

Reviewed operation hashes:

| Path | SHA-256 |
| --- | --- |
| scripts/consolidate_go2_training_artifact_duplicates_v1.py | 86f7074fedf1725458f7519187936afdea14dd2a76d7443e9a9480daf01e00d5 |
| lewm/tests/test_training_artifact_hardlink_consolidation_development.py | db65976d9924f42946d920582e5fb1f270dd6794f65f2af99e743a80a976e56b |
| docs/go2_training_artifact_hardlink_consolidation_v1_2026-09-09.md | 3f4126ad5e45a75d54b7ea0aa84ef20bd54ea603c6156520990a6c6ca1178d2e |

After explicit approval, bind the user's instruction and these identities in
docs/go2_training_artifact_hardlink_authorization_2026-09-09.json, then execute
the reviewed runner with --approval-json naming that file. The exclusive journal
root is go2_training_artifact_hardlink_consolidation_v1_attempt_001. This review
is not authorization. No consolidation or deletion has occurred. Pip cache,
GSD cache, unlisted files, directories and other datasets are excluded.
