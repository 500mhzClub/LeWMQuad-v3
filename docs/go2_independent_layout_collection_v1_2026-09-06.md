# Independent-layout pulse collection V1: fixed per-layout batches

Execute the already frozen inventory only. Inventory SHA256:
`714161041c6db96270d91b53749a982542a5b32fbbee155ec0d5ab98d7afd426`.
Its12layouts,6/3/3train/selection/development-evaluation roles, five contexts,
quiet/recent histories, two support conditions, six action cells and all1,440
episode definitions are immutable. This is development collection, not final
evaluation, learned navigation, real-platform validation or permission to bypass
hardware safety. The prior pilot and failed/corrected audits remain unchanged.

## Execution unit and invariant construction

There are twelve whitelisted batches,l00–l11, each the exact120episodes of one
layout in inventory order. Each batch has its own owned RecoveryStorage root:
`go2_independent_layout_collection_v1_lXX_attempt_001`. No arbitrary episode list,
replacement spawn, role change, implicit retry or resume is supported. Start with
l00; audit its attempted/committed evidence before deciding whether implementation
issues require a distinct successor. Do not silently alter later layout definitions
or hide failed near-wall contexts. Additional batches retain the declared roles.

Bind the full scene adapter, session, batch driver, auditor, tests and protocol,
the fixed inventory and inherited native/gait/calibration identities before any
physics. CPU only, one native environment, frozen low-level walking policy and
gains. The distinct inventory initializer participates in the tested inherited
RGB/body/depth/fast-gyro/native-contact recorder chain; no old initializer is
monkeypatched or edited. Exact native wall boxes must match each specification.

At each declared spawn, record15settling ticks (750physics samples), then check
actual full articulated setup clearance, foot geometry, native support and
velocity. Use the configuration-derived4cm-padded setup envelope, not the old
room-sized empty prism. Native state is evaluator-only. Record eight common zero
or0.12m/s forward ticks, as specified by quiet/recent history. At2.3s/tick8 apply
the assigned2/5-tick pulse, then20zero ticks. Maximum33post-setup commands,
2,400native samples and34paired frames per episode. No post-stop zero drain.

The selector accepts action index, history kind, tick and a causal RGB/body packet.
No geometry, friction, native pose or future target is supplied. Visual-led
RGB-D/gyro tracking is shadow-only and never gates these simulation commands.
Retain disallowed-contact/body-stability stops plus nonfoot-ground,0.3m/s speed
and±8m domain guards. Packet-contract/storage stops remain distinct from collision.
One physical or acquisition stop does not skip the other planned action siblings.
An infrastructure exception terminates the batch without another episode attempt.

## Storage and durable accounting

Require40GiB external free reserve plus an8GiB batch allowance before launch.
Before each episode require another256MiB within that allowance and above the
free reserve. Check free space again at decisions. After each episode, persist
all native/sensor/decision data and bind only its explicit expected artifact list.
Write a new exclusive `episode_NNN_commit.json` containing its result, artifact
hashes, actual byte count and any missing expected paths. Never recursively scan
the artifact root. Record physical-stop episodes too; missing artifacts or an
episode exceeding256MiB terminate the batch after retaining its partial commit.
These are resource checks, not a hardware memory/safety certificate.

After each complete artifact commit, run the offline raw episode reconstruction
before starting the next episode. Persist a separate exclusive raw-precheck report,
prefix witness and any actual window/targets. This consumes no additional physics
and never feeds targets/native state to the selector. A reconstruction error stops
the batch as an infrastructure failure, retaining the committed episode; physical
failure and absent targets alone do not stop the batch. The terminal batch audit
independently repeats each reconstruction and checks exact agreement with every
saved precheck. Keep precheck hashes/bytes in terminal accounting.

The final result, or terminal infrastructure-failure record, binds every completed
commit and identifies any uncommitted in-progress episode. Do not claim incomplete
raw artifacts are audited training data. Preserve all existing evidence; no
cleanup, relocation, whole-tree materialization, sealed access or source export.

## Raw audit and target integrity

Audit only a terminal batch with exactly one result/failure record. Verify source,
inventory, episode construction, committed artifact paths/hashes/byte counts and
planned-prefix ordering. Report unattempted, attempted-uncommitted, incomplete
artifact and fully audited cases separately. Never turn absence into a collision-
free target or silently exclude it from the120planned attempts.

Reconstruct every available raw sensor/contact sample, actual camera pose and
RGB/depth packet, setup envelope/support check, selected command and shadow
decision. Recompute the first native stop and prohibit post-stop physics. Compare
recorded requested commands exactly as float64, including quiet history; retain
the separate applied-command float32/slew check. The previous audit's mistaken
float32-rounding of requests must not recur. Initial yaw checks use the declared
cardinal spawn orientation with wrap-safe differences, not an assumed zero yaw.

Compare all1,150native/contact prefix samples and nine complete sensor packets
within each layout/context/history/support group, against its declared action0
reference. There are20groups and100non-reference comparisons per full batch.
Report missing or unequal prefixes; matching seeds are not actual equivalence.
Derive a pulse window only from an actual departure. Keep exact2.2/2.5s endpoint
times, native contact positives before interruption, censored missing/noncontact
stops and future-image availability distinct. Materialize available windows using
the existing dataset interface with their original layout role. Do not fit here.

Near-wall contexts are hazard hypotheses, not manufactured positives. All60
pilot targets were negative; new native setup/warm-up and contact evidence is
required. Report all action/history/support/context coverage and tracking
availability even if the intended hazard cases fail or produce no contacts.
The camera remains body-hidden, sensing ideal and physics paused during compute.
The raw near-depth comparator does not fill invalid public sensor depths.

The pre-launch optical correction uses a distinct0.005m render near plane,
unchanged rigid mount/intrinsics and unchanged0.2–5m public depth validity. The
completed [native camera bench](go2_near_field_visibility_probe_v1_2026-09-06.md)
reproduces the predecessor's close-wall see-through defect and tests the new
setting, including residual below-near negative controls and far-depth precision.
This changes neither the frozen layout inventory nor its episode definitions;
the unfrozen scene adapter explicitly binds its new render calibration.
The new sampled physical first-surface reference never discards close occluders.
A visibility failure is retained in the raw precheck and terminates the batch
before another episode. Terminal audit retains its physical labels/report but
does not materialize that episode's window as usable learning data. Report such
excluded trials explicitly, not as successful episodes or censored contacts.

## Downstream scientific requirements

After adequate independent collected data, test action/history/RGB utility and
matched direct, supervised-rollout and JEPA objectives under equal data/labels/
seeds/budgets. Then establish reliable local execution and compare online rollout
and memory/backtracking in complete novel-maze tasks. Development evaluation is
not sealed final qualification. Sensor realism, deadlines, complete physical body
sweeps and bounded hardware evidence remain separate required stages. No batch
completion or passing audit establishes the full scientific goal.
