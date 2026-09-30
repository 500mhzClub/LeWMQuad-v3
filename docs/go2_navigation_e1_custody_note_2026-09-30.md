# Custody note: sealed-directory metadata access, 29–30 September 2026

This note is recorded under `AGENTS.md` ("If any command may have read protected bytes, stop immediately and record…"). **No file contents under a sealed path were read.** Two borderline accesses are disclosed for Andrew's judgement.

**Protected path concerned:** `<capability artifact root>/sets/sealed_test_v1`. It is a `sealed_*` directory holding the E1 sealed mazes and episode packets.

## 1. Directory-metadata traversal by `du` (storage surveys)

- **Commands:**
  - `du -s --block-size=1M <artifact root>/*` inside `scripts/survey_go2_e1_storage_candidates_development.py` (29 September, about 23:40);
  - `du -s --block-size=1M *` run inside the capability artifact root (29 September, about 23:35).
- **What was accessed:** directory entries and file sizes (`stat`) under every subdirectory, including `sets/sealed_test_v1`. No file contents were opened.
- **What was output:** only aggregate directory totals, such as "go2_navigation_capability_v1_attempt_001 33.1 GiB" and "sets" within the root's total. No sealed filename, count or per-file size was printed or stored separately.
- **Recipients:** this session's transcript and `e1_storage/candidates.json` (aggregate sizes per top-level directory).
- **Assessment:** metadata traversal only; no protected bytes read. Future size surveys will exclude `**/sealed_*/**` explicitly.

## 2. In-memory reconstruction for set exclusion (registries `c3v2_sets_v1`, `c3v3_sets_v1`)

- **Commands:** `scripts/register_go2_c3v2_sets_development.py` (29 September) and `scripts/register_go2_c3v3_sets_development.py` (30 September).
- **What happened:**
  - Both regenerated all 90 capability layouts, including the 60 sealed ones, **in memory from the generator and seeds**. Each was compared by SHA-256 with the opaque hashes in the public `registry.json`, and its graph was excluded from new construction.
  - No file under `sets/sealed_test_v1` was opened. The regenerated contents were never written, printed or returned; only "verified 90/90" and exclusion counts were output.
- **Status:** this method was declared in the C3-v2 pre-declaration and approved. It is disclosed here because `AGENTS.md` sets a stricter custody standard for final-test material.
- **Andrew's call:** whether future registrations should avoid regenerating sealed layouts altogether, for example by excluding by opaque hash only.

## 3. Reads of other files with "sealed" in their names

- `docs/lewm_go2_v4_sealed_invalidation_2026-07-10.md`: only its first 12 lines were read, to confirm that "V4" in `AGENTS.md` means the legacy Go2 generalization V4.
- This is a tracked incident record, not a sealed file or directory.

## Actions

- Nothing sealed has been accessed for E1.
- E1's sealed arms wait for the custody arrangement proposed in [the launcher proposal](go2_navigation_e1_sealed_custody_launcher_proposal_2026-09-30.md).
