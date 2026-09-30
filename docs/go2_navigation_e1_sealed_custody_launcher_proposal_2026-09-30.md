# E1 sealed-set custody: one-shot launcher proposal, 30 September 2026

**This is a proposal for Andrew's approval. Nothing touches the sealed set until it is approved and set up.** It meets the repository custody rules (`AGENTS.md`):
- the model-facing process never opens, reads or searches a `sealed_*` directory;
- final-test custody needs OS isolation and a fail-closed, reviewed, frozen, one-shot launcher.

It also covers the capability pre-registration's `sealed_custody` record ("no runtime or rendering", "no E1 execution authority"), which Andrew's E1 confirmation of 30 September now supersedes for E1 only.

## The problem

- **The E1 episodes live in a `sealed_*` directory.** The 60 sealed mazes × 2 reserved episodes are at `<capability root>/sets/sealed_test_v1`, readable by the model-facing account (`andrewknowles`).
- **The frozen run owner copies sealed content into every mission's output.** It writes the scene specification (`specification.json`, which is sealed geometry), the episode packet, physics traces, maps and planning logs into `runs/<assignment>/`.
- **Every analysis tool reads those records.** That includes readers, diagnosis rules and replays.
- **So running E1 the way qualification and the checks ran would put sealed bytes into model-readable outputs.**

## Proposed arrangement

**1. Custodian account (Andrew sets up; this needs sudo, and I will not do it).**
- Create an OS account, for example `lewm-e1-custodian`, with no login shell.
- Move `sets/sealed_test_v1` into a custodian-owned root, for example `/srv/lewm_e1_custody/`, with mode 0700 owned by the custodian.
- After that, the model-facing account cannot read it.

**2. A frozen one-shot launcher,** written by me, reviewed by Andrew and committed. Its SHA-256 is bound in an authority file that Andrew commits.
- A single sudoers rule lets the model-facing account run exactly `sudo -u lewm-e1-custodian <python> <launcher> --authority <path>`, with fixed arguments.
- **What the launcher does:**
  1. Verifies its own SHA-256 and the frozen harness, model and code bindings against the authority file. It fails closed on any mismatch.
  2. Verifies every sealed packet's SHA-256 against the opaque hashes in the public registry.
  3. Runs each mission with the frozen owner, swapping only the episode loader (the sealed packets), the model loader (version and seed) and `Budget` (the E1 running-time budget, bitwise-replay verified).
  4. Writes every raw per-mission output to a custodian-owned output root, with no copy to the model-facing roots.
  5. Runs the frozen readers and the frozen mechanism-diagnosis rules inside that root.
  6. Exports only a **sanitised results file** in a declared schema to a model-readable export directory: per mission, the assignment, controller, version and seed, and the pre-registered outcome metrics. Geometry, poses, maps, frames, traces and packet contents are never exported.
  7. Exports sanitised progress lines (assignment identifier, status, elapsed time) and budget and storage status.
  8. Stops and exits automatically at each pre-declared stop: the pilot and seed boundaries with re-projection, the running-time caps (primary 157 h; exploratory arm per its declaration), the storage stop (at least 15 GiB above the reserve), and safety violations. **Continuing needs a new authority file from Andrew.**
  9. Is one-shot. It records a marker when an authority is used and refuses a second run of the same authority. Failures are preserved, with no automatic retry.
- **What the export contains per mission:**
  - beacon, home and round-trip success; SPL (outbound and return);
  - time to beacon and to home; disallowed contacts; hard and operating violations;
  - selected-hold fraction per leg;
  - the frozen mechanism label, latch-active fraction and pose-loss flag for failures;
  - decision latency; wall time.

**3. Testing without sealed access.**
- The launcher has a `--rehearsal` mode that runs the same code on a development registry: the unused C3-v3 round safety-check mazes after the exploratory safety check, or fresh development mazes.
- It runs as the model-facing account, and its outputs are checked against the normal pipeline.
- Andrew reviews the rehearsal outputs and the export schema before freezing.

**4. Analysis and report.**
- The E1 report is written from the sanitised export only.
- Paired per-maze differences, the maze-cluster bootstrap, mechanism tables and per-seed results all come from export rows.
- Exploratory-arm results are kept in a separate section and separate tables.

## What Andrew needs to decide or do

1. **Approve this arrangement**, or specify another.
2. **Set up the custodian account, move the sealed directory, and add the sudoers rule.** I will not run sudo or change ownership myself; those are the custodian's steps.
3. **Review and commit the launcher's authority file** after the rehearsal.

## Estimated timing once approved

About 2–3 h to write the launcher and its rehearsal on development mazes, then Andrew's review. E1 itself then runs as projected: primary arm about 131 h (cap 157 h), plus the exploratory arm under its own cap.
