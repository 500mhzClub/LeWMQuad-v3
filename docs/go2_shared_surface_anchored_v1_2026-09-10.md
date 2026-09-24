# Invocation-local surface receipt sharing

The completed anchored-controller profile identified repeated copies of detailed
surface receipts as a major cost. The separately tested faster recursive copier
reduced total paired replay time by only 2.1%; batching floor queries reduced it
by 4.95%. All measured post-warmup calls still exceeded 100 ms. This candidate
removes repeated copies of one read-only field during the original recovery
calculation. It does not combine either prior optimization.

Keep the exact code objects of the six previously reviewed functions: nominal
constraint, eight-step planning, surface filtering, hold reconsideration,
anchored hold reconsideration and its dispatch. Private namespaces substitute
only their copy provider and internal references to those six functions.
Imported module globals remain unchanged. These functions read surface_checks
and may replace that field, but never mutate its descendants; candidate rows
remain independently copied by ordinary deepcopy and all original calculations
and gates execute.

One workspace exists for one selector invocation. For each candidate copy,
accept sharing only for a built-in list/dict surface graph with ordinary scalar
leaves and string keys, without cycles, whose containers are disjoint from every
other selection field. Other selection fields must also form an ordinary
acyclic graph. Shared descendants within the surface graph retain their aliasing.
Cross-field aliases, custom values, unsupported containers and cycles use the
original deepcopy. Strong references prevent cached object-identity reuse.

The only skipped copy is the surface_checks root, seeded into the ordinary
deepcopy memo. All other fields use its normal copying behavior. A changed
public result detaches borrowed surface roots with ordinary deepcopy and a
common memo, preserving aliases and all receipt values. Non-borrowed custom
leaves already have the original calculation's ownership and are not copied
again. Returning the original input by identity remains unchanged when no
recovery applies. No workspace is retained between observations.

The controller inherits the original mapper, memory, model, residual tracker,
observation, mission and command execution. It changes only the selector's
recovery helper and two explicit controller metadata fields. Preserve complete
predictions, selection receipts, commands, observed state and model weights.
Do not change, restart or adopt this implementation in a currently frozen job.

Before a timing or behavior-equivalence claim, test original recovery outcomes,
every existing gate and exception, alias fallback, custom copying and public
mutation isolation. Then prospectively replay the original 405-observation
JEPA prefix with two fresh models/controllers, compare complete normalized
decisions and observed state, and measure the same fixed windows without
profiling. Test success alone establishes neither a speedup nor native behavior.
Navigation, independent layouts, full sensing/control timing and real-platform
evidence remain separate unfulfilled requirements.
