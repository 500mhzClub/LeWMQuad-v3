# Native planning-map persistence comparison preparation

Prepared, not executed. A completed current learned maze0 baseline and the
already completed eleven-observation planning-map prefix are required. The
current native baseline is still auditing; no new native output was created.

The new collector and raw audit are derivatives of the current baseline with
the controller class and explicit planning-map metadata changed. Physics,
sensor acquisition, renderer witnesses, low-level gait, raw controller/command
reconstruction and independent arrival/retracing/quiet/visibility calculations
remain unchanged. The worker loads the same assigned model for collection and
a fresh identical model for raw replay. Collected artifacts and completed audits
remain bound if a subsequent check fails.

The native prefix checker admits the exact completed planning-memory result
and all eleven saved decisions. It compares1250physical samples and eleven
paired public packets against the completed current baseline, then requires
all eleven new decisions to match the prospective replay. Ten earlier actual
requests, shared observed/executed residual state and eight raw forecast banks
must agree. At10the baseline request is[0,0,.45]and candidate[.16,0,.45].
No outcome after that command is compared or borrowed; a partial new command
caused by a physical stop remains a possible negative execution outcome.

The launcher requires actual --learned-result-sha256 and supports
--preflight-only. It preserves one CPU scene/process/numerical thread, the
3000tick shared mission budget,32GiBRAMadmission and10GiBcollection plus1GiB
persistence above40GiBreserve. No retry, overwrite or automatic adoption.

Validation:

- Tests6876closed25pass2.57s: complete saved-prefix admission, invalid/missing
  decisions and identities, physical/public/shared-state mismatches, changed
  prior/new requests, truncated physical prefixes, no comparison after1250,
  partial new-command handling, AST equality of original physical/audit
  calculations outside the explicitly enumerated controller/metadata changes,
  fresh model loading and preserved artifacts after audit/prefix/verification
  failures.
- CLI85267closed successfully.
- Source preparation59039closed1678prepared native paths. All1670frozen prefix
  sources, including1654live baseline sources, checked and unchanged. The actual
  completed prefix admitted all11decisions with first command difference10.
  Native output root absent; no native preflight, scene or checkpoint load.

New source SHA-256 identities:

- scripts/current_observation_planning_maze_episode_development.py:
  101c533b49ce4d803e06f0cde20da0ab54a3431ec7c6388087f422d518be5c3e
- scripts/current_observation_planning_maze_audit_development.py:
  8bae809387e33c00f5afbe4822933358c65978dc5d3a6469accf0d0a0a1c0b7a
- scripts/current_observation_planning_native_prefix_development.py:
  c85ab5d46904990be1d12f6a18073a24601cb91f830d991b908df73402a1ab39
- scripts/run_go2_current_observation_planning_maze_pilot_v1.py:
  c854388baf32eb1bd6bb98c5d732fe691203e9b0149ce606dff1b529f2f10370
- lewm/tests/test_current_observation_planning_native_development.py:
  250ac41ec6adb832c03ffcc8ced86a9bfc8dde51c9ca164dda5fbae2f314d5c7
- docs/go2_current_observation_planning_maze_pilot_v1_2026-09-09.md:
  328c4bb2aa7c74c5b75087ab97df9aaf4c4c1d708de3676edeea3fff74eff918

Retain the queued order: current baseline audit/readout, independent learned
layouts1–3, reactive maze0 and paired readout, independent reactive layouts1–3,
then this planning-memory native pilot with fresh hardware assessment. Its
physical result and a paired outcome readout remain outstanding. No memory
advantage, new arrival, round trip, real-time or hardware claim is established.

Current audit process2447506confirmedactiveRl99.1%CPU after106m25s,
CPU105m30s,RSS10,683,760KiB. Native19976has no root result/failure, case audit
or worker terminal yet. Continue the same process; no restart or source change.
Goal active/unachieved,zero verified round trips.
