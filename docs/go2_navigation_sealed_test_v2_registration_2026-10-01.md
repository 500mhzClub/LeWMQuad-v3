# Sealed test set v2: registration (1 October 2026)

**Why.** Andrew declassified the capability test set (layouts 30–89, formerly `sets/sealed_test_v1`) for the development-mode preliminary run. It is renamed `sets/prelim_test_v1` under the AGENTS.md preliminary-test exception, commit 657ca4c8. The rigorous phase needs a set nobody has seen, so a fresh one was generated.

**How** (`scripts/generate_go2_sealed_test_v2_development.py`, commit 50132f1d):
- **Same generator as the capability sets:** `independent_round_trip_layouts_development.build_inventory` with the capability `make_spec` and `episode` rules, two episodes per maze.
- **Same exclusions, extended to every maze built since.** 257 excluded graphs in all:
  - the capability prior graphs;
  - all 90 capability layouts (development, validation and prelim_test_v1, each hash-verified against the capability registry);
  - the fresh-check set (`c3v2_sets_v1`);
  - the C3-v3 round set (`c3v3_sets_v1`).
- **Candidates rejected** if their abstract topology or grid embedding matches any excluded graph. 128 candidates were examined.
- **Seeds:** the construction seed and the physics, appearance and episode seed bases were drawn from `os.urandom` inside the generating process. They were written only inside the sealed folder and never printed, so the model-facing account cannot regenerate the set.
- **Structural checks only.** No physics, rendering or model.

**Result** (public receipt `<capability root>/sealed_test_v2_receipt.json`, sha256 `45460286107d3af79ee231938789533d013c784907b372600476c00ea9bd37c5`, 28,534 bytes):
- **Files:** 60 mazes and 120 episodes (182 files, including the sealed registry and construction evidence). Hash readback matches.
- **Structural checks, all passed:**
  - 60 contiguous layouts;
  - 60 unique abstract topologies and 60 unique grid embeddings;
  - disjoint from all 257 excluded graphs;
  - 2 episodes per maze, each meeting the occlusion and endpoint-clearance rules;
  - roles `sealed_test_v2`;
  - no physics or rendering.

**Custody.** `<capability root>/sets/sealed_test_v2` is a `sealed_*` directory. The AGENTS.md sealed rules apply, and the model-facing account never opens, parses or searches it. It stays untouched until the rigorous phase, which also needs the operating-system isolation and one-shot launcher that AGENTS.md requires for final-test custody.

## Seeds and backup (1 October 2026, Andrew's follow-up)

- **Seeds are already inside the sealed folder.** The generating process wrote the construction seed and the three seed bases into the sealed registry, `sets/sealed_test_v2/registry.json` (key `seeds`), alongside the mazes. That file is hash-bound in the public receipt (registry sha256 recorded there). A separate seeds file was not created: it would require the model-facing account to read the sealed registry, which AGENTS.md forbids. The seeds remain unseen until the rigorous phase, when they can be published from the registry with the paper.
- **Backup off RecoveryStorage.** A byte-for-byte copy is at `/mnt/workspace_drive/LeWMQuad-v3_sealed_backups/sealed_test_v2`, on the workspace NVMe (nvme0n1), a separate physical disk from RecoveryStorage (encrypted root disk, nvme1n1), outside the repository. It keeps the `sealed_` name, so the AGENTS.md sealed rules apply to it.
  - Made by `scripts/backup_go2_sealed_test_v2_development.py` (commit 2392b7af).
  - Verified by hash with nothing displayed:
    - 182 files;
    - all 180 maze and episode files and the sealed registry match the public receipt;
    - the construction evidence matches the source;
    - the source still matches the receipt.
  - Backup receipt: `<capability root>/sealed_test_v2_backup_receipt.json`, sha256 `801407c55a488516da264aaacd85a0f161cf3fedef3f151553cbfef16471ded5`.
