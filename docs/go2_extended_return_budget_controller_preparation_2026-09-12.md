# Longer-budget retained memory and controller composition tested

The longer mission and transport components are now connected to a fresh
chained single-pass controller composition. No native pipeline or running
controller instance has been changed. The prospective mission permits at most
8,000 navigation steps and 8,014 total observations; a fresh native execution
still requires recording/evaluator integration, resource admission and actual
recorded-prefix evidence.

## Source identities

| File | SHA-256 |
| --- | --- |
| `lewm/extended_return_budget_memory_development.py` | `f18a8eaedf8d8a43192d09de91f9bad41502b3fe1fafa96d43081d938969bcab` |
| `lewm/tests/test_extended_return_budget_memory_development.py` | `11e3b19f483e8633c9eda71591fbff04afcec9c752213b48a375e475b575986d` |
| `lewm/extended_return_budget_controller_development.py` | `fe4b7bd3268e53f26f2933492d0d5b5417a28dba130ccf02fae10f14770c5884` |
| `lewm/tests/test_extended_return_budget_controller_development.py` | `9e1a81eaf3099d6984e063706dbea09654b626ff3c6e1e5c336dc326fc967da1` |

## Retained memory and geometry

The surface observation and paired later-floor ledger use private bindings
with the 8,014-observation ceiling. The surface and residual consumers use the
extended measured-floor pose accessor. The body-projected patch append method
changes only its literal frame ceiling; all measured classification and prefix
arithmetic remain identical. The recording geometry retains tiled indexing and
paired camera coverage while dispatching to that append method.

The memory component passed **24 tests in 6.44 seconds**, tool session 93326.
The compact later-floor ledger was exercised through every frame 0–8013,
retaining all 16,028 ordered primary/auxiliary camera records. Its last view can
resolve an earlier ambiguous sample, but cannot resolve a same-frame sample;
the next out-of-budget pair is rejected without publication.

Separate surface and image-history tests exercise old and new boundary frames,
preserve earlier entries, compare exact current geometry/prefix bytes, and
reject capacity, clock, pose, hash, floor or depth errors. The surface failure
latch and immutable patch prefix remain. These tests pad earlier large histories
with metadata placeholders: they do not allocate or validate a complete
8,014-frame raster history or establish its memory/resource envelope. The
residual test checks the admitted late transport pose without inferring an
executed command or observed residual label.

## Fresh controller integration

`ExtendedReturnBudgetChainedController` first constructs the original empty
optimized components using an originally admissible temporary mission. Before
returning from initialization, it privately composes the extended mapper,
surface memory, paired ledger, registration, residual and selector and installs
the actual extended mission. It rejects noninteger/out-of-range budgets and
does not upgrade an already observed or already converted instance.

The original eight distinct empty single-pass bound indices and progressive
patch histories are retained. Mapper/surface and selector/residual aliases are
reconnected consistently. The chained visual-motion object is unchanged.
Both the optimized selector's exact memory-type check and the receipt-view
type check bind the new memory class; its read-only query view still shares
the original field dictionary. Controller advancement uses the same extended
pose accessor as registration, surface memory and residual processing.

The controller passed **10 tests in 42.35 seconds**, tool session 50532. These
include fresh-state/alias/index checks and two short image-to-action sequences
for direct/no-RGB and JEPA/full conditions. They use independent fixed-head
test models, real image/depth tracking and articulated geometry, compare every
complete decision and retained controller field, and verify geometry and model
state separately. Only the seven explicitly listed implementation type tags
are normalized in retained state; decision normalization removes the new
implementation declaration and restores the predecessor controller identity.
The optimized footprint scope is observed to execute. Duplicate sensor input
still latches zero-command failure, and test model weights remain unchanged
with no gradients.

Those short comparisons use the same 40-step mission allowance in both arms.
They establish composition behavior at that allowance, not full recorded
history equivalence with a changed 8,000-step allowance. The earlier mission
tests separately cover the declared-budget difference and late deadline rules.

The initial controller test run, tool session 53733, had **two failures and
eight passes in 10.40 seconds**. It stopped at the test's geometry fingerprint,
which passed a link-name set directly to a helper that only accepts JSON-like
containers and arrays. The correction routes geometry through the existing
structured-state serializer before fingerprinting. No controller change or
decision/state comparison was weakened to resolve that test-harness failure.

All tests used the original deterministic single-thread environment and
`pytest -q -p no:cacheprovider`. They overlapped the already declared non-isolated
chained/single-pass timing comparison. Afterward, all 2,639 sources bound by
that live comparison were independently rehashed and matched. The four new
files are outside its source roster.

## Remaining before the extended native trial

Complete and authenticate the running 4,014-observation controller comparison.
Then verify the extended controller on the original recorded prefix, accounting
only for declared budget and implementation changes and stopping at the first
prospective decision intervention. A passing short synthetic sequence cannot
replace that check.

The native collector, complete decision stream, sensor/renderer witnesses,
public packet/replay bounds, command audit and evaluator still need the same
extended population and endpoint limits. Resource checks must cover the full
retained raster history, raw persistence and audit population. No old terminal
episode may be resumed or relabelled as a successful longer run.

No new native launcher, queued scene, acquisition optimization adoption,
independent-maze execution, completed physical return, real-time result or
hardware qualification is established here. The earlier implementation map is
`docs/go2_extended_return_budget_preparation_2026-09-12.md`; its memory,
transport and controller items are now implemented at the component level,
with the validation limits above.
