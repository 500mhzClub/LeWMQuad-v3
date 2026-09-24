# Matched JEPA and supervised-rollout causal prefixes V1

Compare the fixed first-seed full-RGB JEPA and supervised-rollout models from
the completed18-fit study on all three completed independent learned mazes1,2,3
in that order. Use the pure matched-objective admission from
`lewm/matched_rollout_objective_admission_development.py`: same initialization,
data, schedule,1200updates and rollout head; only the declared training objective
differs. Preserve each arm's own training-only intercepts fitted using the same
estimator, training population and weighting. This compares the two fitted
pipelines; it does not force unequal learned correction values to be equal.

Authenticate the original learned cohort result
`a0ac72330a6c34fbae7812a360b22189086f16bd83f586d71e769539ddc2e720`,
all its bound artifacts/upstream sources, both exact fit metadata files and the
existing complete fit/correction admission. Load each assigned model through
the existing evaluation-only loader. Verify exact base states and correction
buffers. JEPA's complete corrected state must equal the original native state;
record the supervised corrected state and require both unchanged at completion,
with no gradients. No training or checkpoint selection.

Use a fresh pair of MeasuredFloorTransportController instances per maze with
identical sensors, geometry, persistent memory, public mission and3000tick
shared budget. Both selectors must use rollout_outcomes. Reconstruct paired
public packets from the actual episode start and pass the exact same arrays to
both arms. Require no mutation. The JEPA decision must reproduce the complete
original decision exactly at every consumed observation. Both arms' raw and
registered pose evidence, mapping and mission receipts, observed goal and
floor-partition state must match. Warmup is completely identical except the
declared model-condition label. Each forecast must contain all6x8x5values,
the original100–800ms clock, the same full-RGB rollout head, and its own exact
training-only XY bias. Preserve all forecasts, scores, vetoes and decisions.

Allow each arm to retain its own online residuals against its own predictions.
Their numerical values need not match: they are causal consequences of the
declared model change under the same executed commands, not newly fitted maze
parameters. Do not reset history or copy the JEPA residual into the other arm.

Stop each case at the first different requested command or terminal decision,
or either arm's terminal. Never read the next recorded observation after that
boundary, even if the original command was physically completed. A raw
prediction difference alone does not terminate a still-common command prefix.
Maximum observations are215,504,265for mazes1,2,3, respectively, ending at each
original first terminal observation. Retain negative outcomes and all fixed
cases; do not substitute models or layouts according to new results. A contract
or source failure halts the attempt and retains already saved candidate rows.

Run `scripts/replay_go2_matched_objective_prefixes_v1.py` using the existing
deterministic single-thread Genesis Python environment. Exclusive output:
`go2_matched_objective_prefixes_v1_attempt_001`. One sequential CPU worker,
12GiBavailableRAM admission,512MiBoutput allowance and existing40GiBartifact
reserve. These are admission/storage checks, not enforced process RAM quotas.
Refresh hardware and competing jobs before submission; at most one separately
owned native scene may coexist. Model-pair states are fresh per case. Recheck
all input/source/artifact identities after all cases. Preflight-only returns
before output creation or model/controller execution.

This is a prospective command comparison on recorded development inputs, not
physical evidence for the alternative trajectories. No simulator, hardware
motion, sealed access, model training, automatic retry or deletion. It leaves
the active planning-memory, queued residual and tracking native order unchanged.
Follow with fresh paired native execution and raw audits before any navigation
attribution. Three layouts with one optimization seed cannot establish seed
robustness, JEPA advantage, statistical reliability or deployment readiness.
