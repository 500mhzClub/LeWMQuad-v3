# Chained native result validation preparation

The prepared native audit path now has a result validator that reconstructs
the actual intervention boundary from raw evidence. It does not use the old
measured-plane pilot's fixed frame-122 intervention. This is source preparation;
no chained native launcher or execution is created by these helpers.

`scripts/measured_plane_chained_native_result_development.py` supplies:

- `prefix_result`: derive availability from the authenticated replay's actual
  boundary and collection counts. If the interval exists, call the existing
  complete physics/public-packet/both-controller prefix reconstruction. Counts
  alone cannot establish a physical intervention. If the interval is absent,
  preserve the precise early-negative reason without reading a missing prefix.
- `require_worker`: require the exact assigned case, worker status, corrected
  model identity, unchanged model, measured-plane estimator, original 4,000-tick
  navigation budget, and bounded observation population. Require the complete
  learned-model raw audit and consistent outcomes. Reconstruct the actual
  prefix and compare the saved receipt exactly, including types and hashes.
  Retain the original joint physical/strict-visibility success criteria and
  reject success claims when the prospective intervention was not completed.

The caller remains responsible for authenticating the completed replay and
complete native artifact rosters, choosing the frozen case and worker status,
admitting hardware, serializing native execution, and preserving failures.
The separately prepared admission helper performs predecessor verification;
the pipeline performs collection and full raw audit; the prefix helper checks
the actual common trajectory. This module connects the resulting evidence to
worker acceptance. It does not grant execution or navigation qualification.

All 21 focused tests passed in 2.41 seconds. They exercise complete synthetic
physics and controller prefixes at three distinct intervention indices, a
terminal-only intervention with unchanged commands, altered physical-prefix
hashes and controller decisions, invalid model/budget/audit evidence, strict
physical and sensing success criteria, early negatives, and required raw
reconstruction when the collection counts cover the interval. These synthetic
tests are software validation, not navigation evidence.

Source preflight verified 2,624 source bindings across the prepared input,
pipeline, prefix, result and test closures, retaining both original waiter
ancestries. The two added source identities are:

| File | SHA-256 |
| --- | --- |
| `scripts/measured_plane_chained_native_result_development.py` | `5e5787cb58f77aaa7e2321f2ca6e3bafd3bc0721bc457a22cdb592fe4a3c967e` |
| `lewm/tests/test_measured_plane_chained_native_result_development.py` | `763114aabe35e7cee3dea50b18b724f47f895183403beb37d3b553d035f88dda` |

At the final live check, the timing replay owner remained running and had
matched complete decisions and unchanged inputs through frame 2276. Both
original waiters remained live. The chained controller replay child root was
still absent. Available memory was about 63 GiB and artifact space about
542 GiB. No live or queued bound source was edited.

The next operational dependency remains the completed timing replay, followed
by the already queued chained-controller replay. If that produces a valid
admitted candidate intervention, prepare and freeze the separate native
launcher/protocol using these helpers, then admit and execute the new run.
A negative replay must remain negative. The full goal remains incomplete.
