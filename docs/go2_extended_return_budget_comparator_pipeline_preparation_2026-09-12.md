# Longer comparator collection and full raw audit prepared

`scripts/extended_return_budget_comparator_pipeline_development.py` now
connects the four prepared comparator modes to collection, complete raw
decision replay, command auditing and phase resource guards. It is a reusable
development pipeline with no launcher, checkpoint assignment or population
roster. The existing single-case longer native launcher is unchanged and
does not use it.

The modes are `frozen_reference`, `nominal`, `reactive` and
`current_planning`, using the components recorded in
`docs/go2_extended_return_budget_comparators_preparation_2026-09-12.md`.
Nominal remains predictive, reactive remains a whole-method comparison, and
current planning removes accumulated routing cells from selection while
retaining the other declared memory. These distinctions are preserved in
the definitions and audit attribution.

## Implementation

The predictive modes retain the original longer learned collection and full
raw-audit function bodies. The reactive mode privately binds the original
reactive collection and full raw-audit bodies to 8,000 navigation steps,
8,013 command ticks, 8,014 observations, the extended packet/stream/renderer
dependencies and the extended native evaluator. Original scene, sensor,
physics, terminal and persistence functions remain in place. No original
function globals or source files are changed.

All four modes use the same checked extended sensor session and sampled
resource guards around acquisition, controller calls and raw sensor audit.
The 28 GiB collection allowance, downstream persistence/audit headroom, RAM
floors and phase accounting remain those of the prepared longer pipeline.
Each guard is finalized after the original function's cleanup. An original
exception or resource breach prevents phase-completion claims while retaining
partial evidence.

Nominal execution surrounds both collection and raw audit with the existing
root-and-child model forward guard. It detects attempted inference even when
controller code catches the guard exception, removes only its own hooks on
exit, and prevents the phase from being marked complete. A successful nominal
audit is explicitly attributed as complete nominal/controller command replay,
not learned-model replay; it records zero learned forward calls. Its reused
command-tape role string remains `online_learned_round_trip_command` because
the original physical and command-audit code is preserved. That historical
format label is not used as evidence that nominal execution called a model.
Every decision's explicit source provenance and the mode-specific audit
provide the treatment attribution.

The reactive interface rejects model, condition and variant arguments before
creating resource or episode files. Predictive interfaces require a supplied
model; authenticating the assigned checkpoint remains the future runner's
job. A future runner must also authenticate closed artifact rosters, verify
matched treatment, provide whole-worker accounting and complete its
prospective experiment admission. This module does not claim those checks
have already occurred.

## Observed verification

Both focused invocations used the existing deterministic single-thread
Genesis Python environment, `-B`, `PYTHONDONTWRITEBYTECODE=1`, and pytest
`-q -p no:cacheprovider` on the exact named file. Both were first observed
invocations and exited zero, with no discarded failed invocation.

- `test_extended_return_budget_comparator_pipeline_development.py`:
  session 47870, **26 passed in 1.96 s**. It checks original code/dependency
  bindings, common budgets, explicit treatment scopes, guarded collection
  and audit dispatch, persistence after failure, no next controller call
  following a resource breach, rejection of invalid modes/model scope, and
  nominal inference rejection for root, child and swallowed calls during
  both phases.
- `test_extended_return_budget_comparator_full_budget_development.py`:
  session 82114, **7 passed in 19.68 s**. All four modes execute the original
  collection and complete raw-decision replay loops with a real extended
  measured mission, synthetic acquisition/controller behavior and synthetic
  physical samples. Each population has 8,014 decisions and camera frames,
  8,013 completed commands, 401,400 physics samples and ten terminal drain
  commands. Frame 4003 is nonterminal; frame 8003 exhausts the shared budget.
  The actual independent command audits pass, and the phase checker
  reconstructs 32,058 collection resource samples and 16,032 audit samples.
  Tests also alter frame 8002's complete decision or command role, or the
  final post-slew command sample; each change fails the original reactive
  raw audit and leaves an incomplete audit resource receipt.

These tests validate bounded orchestration, persistence, replay and command
accounting on synthetic inputs. They do not execute native physics, verify
actual sensor reconstruction, measure learned inference performance or
demonstrate navigation. The separately recorded controller tests cover short
actual-image processing. Neither substitutes for prospective native trials.

Source-only discovery/check, session 26267, exited zero with a 2,682-path
union: the prepared native ancestry, four comparator component/test paths,
and these three new pipeline/test paths. It reverified all 2,639 source
bindings of the active original full-history replay as unchanged. The
budget-prefix and longer native attempt directories were still absent.

| Exact path | SHA-256 |
| --- | --- |
| `scripts/extended_return_budget_comparator_pipeline_development.py` | `7873edb070151bdde3919367bfc638d3240c9ccd79df4967b2eaa06a0083a629` |
| `lewm/tests/test_extended_return_budget_comparator_pipeline_development.py` | `8616144567712a2940e36fcb6613b68e734ff31cb4350a5285b9890e29220c76` |
| `lewm/tests/test_extended_return_budget_comparator_full_budget_development.py` | `9b1ec3f8cef62000561e571b207c99af22c6a090e5c90183df4041982ad6f9b4` |

## Live execution and next step

At the recorded observation, the exact original comparison owner PID 3015121,
creation time 1789209876.83, remained live with 2,968 of 4,014 observations
completed, matching original/normalized candidate decisions, and no terminal
result or failure. Short synthetic tests overlapped it; its timing remains
nonisolated development evidence. Its launch SHA is
`b8a67fc80aebb2f39968092067cea00b94304c449bd7c86fdff30458c58d6260`.

Continue the actual evidence sequence: finish and authenticate that comparison,
run the prepared budget-prefix replay with its actual completed result SHA,
then the prepared longer native trial only after a positive prefix. Preserve
negative outcomes. These comparator pipelines will need prospective native
admission and independent-layout integration after the actual longer result
is reviewed; no population is silently admitted here. The goal remains
incomplete, including independent generalization, matched JEPA/planning/memory
evidence, realistic continuous timing and bounded hardware evidence.
