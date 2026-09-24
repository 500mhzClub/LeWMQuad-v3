# Supervised commitment-contact prefix completed

Replay93799/PID2568108 exited0 with status
SUPERVISED_COMMITMENT_CONTACT_PREFIX_V1_COMPLETE. Exclusive artifact root:
`go2_supervised_commitment_contact_prefix_v1_attempt_001`.

| Artifact | SHA-256 |
| --- | --- |
| result.json | 37b29828635e88fab77f81447f6a05b890911426fc3478d8aac451426a229de0 |
| launch.json | a412b59f4f865daf8920bd1a3f894a4fa3f685985c4d1130c1ea38cc19407e18 |
| context_decisions.jsonl.gz | deedbdd7ae5840c632c72495d0797011bad851506e58a05780bd56946dff8735 |

The four-observation actual prefix reaches frame3 with three identical prior
commands and one exactly matched complete six-action forecast bank. At frame3,
original supervised right turn[0,0,-.45] becomes right arc[.16,0,-.45]. The only
assigned policy change is intermediate-waypoint contact-cost horizon800ms to100ms;
pose utility remains100ms, contact coefficient remains1.2 and every original
800ms path, surface and phase constraint remains exact. Observed-state receipts,
model forecasts, model state and public input arrays agree. No terminal/failure
occurs in the candidate prefix. No frame4 observation is consumed.

The inherited first-step geometric potential progress for right arc is
.013268335642943874m. Its retained800ms contact score is.006997195675243411,
giving old utility.004871700832651793m. Its100ms score is.0001236472229186693,
giving new utility.013119958975441482m. Right turn's old utility is
.006193184799936241m and new utility.007216563819274523m. Thus the contact
horizon alone reverses their ranking at the first selection. Forward also gains
utility but does not win. Neither contact score is a calibrated probability.

Preflight43825 and actual pre/post verification passed. All1721 sources and
original transitive verification conditions were authenticated. Each actual
verification used944,482 digest requests and137,864 unique files, with
64,214,606,916 bytes hashed initially and freshly again finally. The benchmarked
helper used23 isolated verifier functions, retained no cache and changed no
imported globals. This is not an atomic-snapshot claim. Reported178.8003472359851s
wall covers work after launch, including final verification; it is not per-frame
controller latency or real-time evidence.

Independent49502 exited0: result identity, all1721 source bindings and output
bindings checked before/after, original completed worker/decision/tape identities
verified, all four complete saved comparisons reexecuted with the frozen
comparator, exact population and boundary summary confirmed. It did not load
and rerun the model or independently repeat the original transitive verifier.
The original replay itself performed fresh-model public-packet execution and
full original input verification before/after. Narrow intermediate inspection
16907 is superseded by this completed result authentication.

Residual equality refers to the complete saved observed-history receipts. The
internal pending forecast for a newly selected command appropriately refers to
that action; neither its unobserved label nor later state after a changed command
is compared. No following physical observations, clearance, arrival, round trip
or navigation benefit can be inferred from this nonphysical replay.

Final hardware:71,530,381,312 bytes available RAM,694,917,005,312 artifact bytes
free,21,357,912,064 workspace bytes free,3.3% CPU busy with full32-core affinity,
card1 GPU0%. Supervised19047/worker2563586 remains live on maze2; latest timing
inspection reached tick1714. Scheduler37343/PID2551088 still waits for the
complete original cohort. No new native scene was independently launched.

Next prepare a separately named supervised commitment-contact maze1 native pilot
using this exact model/controller and fixed first intervention. Require fresh
public/physics prefix equivalence through frame3, completed changed command,
complete raw controller replay and existing native arrival/round-trip/visibility
checks. Schedule only after the current fixed native queue completes, without
editing that queue or any running source. Preserve both negative and positive
outcomes. Aggregate remains24 raw-audited native episodes and zero verified
round trips; the overall goal remains active.
