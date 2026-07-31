# V18 spatial-token delay-line Joint-JEPA V1 — update-zero gate-timing integrity replacement V1

Date: 2026-07-31

Status: preregistered, source-only, execution denied pending implementation review, a certified narrow export, and a fresh one-shot authority.

## Reason for the replacement

The original one-shot attempt terminated at update 0 with status
`FAIL_UPDATE0_INTEGRITY_TERMINAL`. It performed zero training updates, zero
presentations, zero optimizer or EMA steps, zero autograd calls, selected no
checkpoint, and created no snapshot.

The terminal observation showed finite, nonzero-scale, hard-synced target,
online, and persistence-initialized memory states. Their global 64-channel
participation effective rank was approximately 1.726, their rank ratio was
approximately 0.02697, and 9.375% of channels were below the registered
near-zero standard-deviation threshold. The controller incorrectly enforced
the absolute update-250 collapse definition at update 0, and would have
enforced it again at update 100.

That timing was not preregistered. The original update-zero gate requires
finite hard-synced states, zero target gradients, zero future-RGB online
access, persistence prediction identity within `1e-5`, valid denominators,
and the inherited place substrate. The absolute participation-rank ratio
`>=0.10` and near-zero fraction `<=0.05` first appear under the registered
update-250 gate.

The intended update-zero facts otherwise passed: maximum prediction versus
persistence delta was `3.635883331298828e-6`; temporal score was one within
numeric tolerance; action and ordered-history lifts were zero; place R@5 was
2.3986874311070916 times exact chance in all eight scenes; target place-key
effective rank was 2.6142773628234863; and target, access, role, and source
custody checks passed. The original result therefore falsifies the implemented
gate timing, not the untrained delay-line mechanism.

## Exactly authorized correction

Authorize exactly one fresh integrity replacement that changes only the
observation/gate-schema adapter and its complete receipts:

- At updates 0 and 100, continue to compute and report target, online, and
  memory participation effective rank, rank ratio, near-zero fraction, RMS,
  finiteness, nonzero scale, and the unchanged `noncollapsed` diagnostic.
- Do not relabel a false absolute `noncollapsed` diagnostic as passing.
- At updates 0 and 100, do not use the absolute rank-ratio or near-zero
  thresholds as a terminal veto.
- At update 0, require finite, nonzero-scale, hard-synced target/online states,
  persistence prediction identity, zero target gradients, zero future-RGB
  online access, valid denominators, and the unchanged place substrate.
- Preserve the immediate update-one gradient-route, optimizer/EMA accounting,
  target-gradient, and finiteness stop.
- First enforce the unchanged absolute target, online, and learned-memory
  noncollapse thresholds at update 250. Preserve them unchanged at updates
  500, 750, and 1000 and in terminal selection.
- Receipts must state whether absolute noncollapse was diagnostic-only or
  enforced at each observation.

No original checkpoint, recovery state, optimizer state, metric artifact, or
random state may initialize the replacement. The original terminal result
remains authoritative and immutable. This is a fresh initialization under a
new output root and one-shot authority, not a retry or resume of the consumed
attempt.

## Science that must remain identical

The replacement must preserve exactly:

- model class and parameterization;
- inherited V18 encoder and representation initialization;
- K4 spatial-token FIFO, local causal depthwise Conv3D reader, action FiLM,
  shared recursive H4 prediction, and EMA target;
- all data identities, train/selection splits, and RGB loaders;
- seed `20260731` and schedule order;
- optimizer, learning rates, gradient routing/scaling, EMA coefficient, and
  initialization;
- physical objective and the full plus 0.5-weight masked-current memory JEPA
  losses;
- 16 memory and 8 physical presentations per update, with one optimizer and
  one EMA step;
- wrong-action, reset, reverse, shuffle, persistence, and HOLD controls;
- observation updates 0, 100, 250, 500, 750, and 1000;
- snapshots at updates 250, 500, 750, and 1000;
- update-250 futility, update-500 continuation, and terminal temporal,
  action, history, place, and physical thresholds;
- stage-A cap 500, terminal cap 1000, memory cap 16,000, physical cap 8,000,
  and combined cap 24,000 presentations;
- no probability-calibration, G2, navigation, held-out, or sealed access.

## Terminal rule

If the unchanged absolute noncollapse gate fails at update 250, terminate and
close this exact delay-line mechanism without another gate-timing replacement.
Passing update 250 does not guarantee continuation: the unchanged update-500
and terminal gates still apply.

## Current authority

This document grants source implementation and CPU-only synthetic-test
authority only. It grants no dataset/RGB payload, GPU, training, checkpoint,
recovery, navigation, probability-calibration, G2, held-out, sealed,
production, promotion, retry, resume, or execution authority.
