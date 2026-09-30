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
