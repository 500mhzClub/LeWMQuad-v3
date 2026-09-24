# Tracking launcher preparation: 354 tests pass; native launch still pending

The draft launcher now connects ordered native workers, raw-tape admission,
paired base replay, all 88 fixed stress streams, post-sensor native scoring and
all 48 predecessor comparisons. It binds successful root output accounting and
retains partial evidence on worker failure without retry or later-trial dispatch.
It requires completion of the original all-12 collection and 36-fit study before
launch. The [draft protocol](go2_independent_tracking_challenge_v1_2026-09-07.md)
defines scope and remaining admission requirements.

The original focused test run had 22 passes and one failure: the successful
final report was incorrectly expected to bind 239 preceding files. There are
238: the 240-path allowance includes an absent failure report and the final
report, which cannot bind itself. The correction also checks the exact actual
root file set, not merely a changed expected number. The corrected focused
launcher suite passed 23 tests.

The resource investigation found that the old complete-tape operation
reservations exceeded the old episode cap. Unfrozen challenge allocations are
now 5 GiB per episode, including 2 GiB deferred recording; the full cohort
ceiling is 52 GiB with an unchanged 40 GiB reserve. Source-only arithmetic
covers the declared operations, but does not establish peak memory or actual
external-writer bounds. No old experiment's configuration was changed.

After this correction, the launcher and collection suites passed 59 tests in
27.60 seconds. The final adjacent regression passed **354 tests across 14
explicit files in 286.79 seconds**, with no failures, errors or skips. This
includes the earlier 330 adjacent tests, 23 launcher tests and one new full-tape
allocation regression. Runtime/test sources were unchanged during that run.
JUnit: `.generated/navigation-development-staging.m6MDz1/independent_tracking_launcher_adjacent_v1.xml`.
SHA-256: `1c11c1ef66cba608cbf845534eac1c43d16bab233ee95fdd0cb51672775d2d87`.

These tests use synthetic native workers/audits plus the existing real
sensor-observer tests. They do not prove native recording, new scene coverage,
tracking adoption, navigation, deployment-valid sensors or a JEPA advantage.

Read-only checks confirm the 701-source completed replay, 771-source live
collector and 786-source planned learning definition remain unchanged. The
original collector has completed six layouts (720 eligible recorded trials),
and the actual `l06` process is live. The latest completed layout retains five
physical terminals and one strict-visibility failure; those are not repaired
or relabeled as navigation successes. Neither the matched-study output nor the
new challenge output exists.

Next: complete the [recording-resource proof](go2_independent_tracking_recording_resource_analysis_2026-09-07.md),
including JSON metadata, native options/overflow behavior and worker/evaluator
peak memory; no successful resource-review file has been issued. Then freeze
the exact challenge definition and launch only after the original learning
study finishes. Independent result verification and fresh closed-loop testing
remain necessary. Latest actual room-return success is still 0/3; the full
scientific goal remains active and unachieved.
