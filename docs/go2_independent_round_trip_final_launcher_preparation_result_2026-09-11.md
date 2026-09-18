# Joined study admission and launcher prepared

The final verifier and CLI are now connected to the existing monitored
one-collector/one-auditor driver. They require the same completed six-case batch,
all five diagnostic waiters, a complete input bundle, and a separately written
policy review bound to actual outcomes and the exact study definition. Runtime
flags alone do not pass admission. No final review or completed input bundle
has been created, and no independent layout has been executed.

The combined suite passed **46 tests in 7.17 seconds**, session 48465, exit zero.
It covers negative scientific outcomes, mixed queue identities, altered models,
budgets, case order, outcomes and prefix findings, actual verifier dispatch,
source/review bindings, rejection of live work, monitored-driver selection,
complete preflight without output, and preservation of failures without retry.
Deterministic test manifests are constructed once and copied for each test.

Source-only preflight passed for **2,198 source paths**, session 72401, exit zero.
It measured 80,233,345,024 bytes available RAM and 610,523,123,712 artifact bytes
free. The original 32-case storage allowance requires 420,906,795,008 bytes;
the staged driver additionally requires 64 GiB available RAM. These resource
measurements grant no policy or execution approval.

[Preparation record](go2_independent_round_trip_final_launcher_preparation_2026-09-11.json):
`018d4c7253d753e38ea424565bedbc8f6936c1ba27afeb6eeec3e4985a87ca24`
(session 42239, exit zero). The preparation confirmed the actual live-budget-
waiter rejection before review/input access, and absence of the bundle, review
and population output.

Once all five diagnostic stages finish, authenticate the input bundle, assess
their complete results, and decide whether to execute or revise the original
study. This launcher accepts only the original 32-case definition and 3,000-tick
budget. It does not select that definition now; changes require a checked
successor. The scientific quality of the written review remains a substantive
review obligation, not something a schema or hash can prove.

Relevant source: `scripts/independent_round_trip_final_admission_development.py`
and `scripts/run_go2_independent_round_trip_population_v1.py`. See the
[execution protocol](go2_independent_round_trip_population_execution_v1_2026-09-11.md).
