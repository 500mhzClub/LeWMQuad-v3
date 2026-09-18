# Approved cleanup requires privileged process inspection

The user explicitly approved the recommended 8,304-file Genesis cache cleanup
with “do it”. Authorization is recorded in
`docs/go2_gsd_cache_retirement_authorization_2026-09-08.json`, SHA-256
`c55dfbe858b7d6e17107189a38c6bc3187afc9c9d2b72b05959a52daa705ca5e`.
The approved proposal remains
`bc405303fb8f7d227c71bf7e965d2262ca6ad8f8e817e78512427bbacf8eaa77`.
Pip cache cleanup is not included.

The unchanged prepared executor ran in session 80367 and exited before output
creation or deletion. psutil.open_files could not read `/proc/3119/fd/0` for the
user systemd process, so its fail-closed process-inspection gate rejected the
operation. A read-only follow-up found the same descriptor-access restriction
for systemd, sd-pam, the document-portal mount helper, ssh-agent and an sshd
session. None of their protected descriptors was read. `sudo -n /usr/bin/id -u`
failed because a password is required. This is an operating-system permission
limitation, not an approval-review rejection or missing cleanup authorization.

The separate privileged executor preserves all original proposal, authorization,
implementation, current-scene, process, metadata, single-link, ownership,
retained-key and per-leaf deletion checks. Only the process inspection runs as
administrator. It explicitly inspects real UID 1000, then permanently drops
supplementary groups, GID and real/effective/saved UID to 1000 before metadata
validation, artifact creation or deletion. The existing frozen retirement helper
therefore still verifies ordinary user-owned files. No inaccessible process is
silently exempted. If administrator inspection also fails, no deletion occurs.

Executor: `scripts/retire_go2_reviewed_geometry_cache_privileged_v1.py`, SHA-256
`c9ecbfae608bafd55f28c78bcf8be3b7baa5156526185cc1ab55cdc07d5306d1`.
Its focused AST scope/order test passed in 0.11 s, checking that all original
logic is retained and all mutation calls follow permanent privilege drop.
No privileged execution has occurred; this test does not establish runtime
administrator availability or successful cleanup.

Run the prepared command in the user's terminal, where sudo can request the
administrator password:

```sh
bash /home/andrewknowles/Workspace/LeWMQuad-v3/scripts/retire_go2_approved_cache_2026-09-08.sh
```

The wrapper first verifies exact proposal, authorization and executor hashes.
It retains the same exclusive output root
`go2_reviewed_geometry_cache_retirement_v1_attempt_001`. After successful cleanup,
verify the journal, candidate absence, retained keys and free space, then resume
the prepared reactive native baseline. All old source hashes remain unchanged;
do not re-request deletion approval or interpret the earlier goal blocker as
still waiting for approval. The current dependency is administrator inspection.
