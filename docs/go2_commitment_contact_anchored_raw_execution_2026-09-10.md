# Commitment-contact anchored raw replay scheduled

At 2026-09-10 13:35:28 UTC, waiter session **6422**, PID **2700744**,
creation epoch **1789047270.48**, is live and waiting for the exact existing
supervised worker PID **2672443** / creation epoch **1789037034.85** under
batch parent **2659758**. The supervised collection is complete but its full
audit and terminal record remain pending. No raw replay child or new native
scene has started.

Waiter root: `go2_commitment_contact_anchored_raw_prefix_wait_v1_attempt_001`.
Launch SHA-256:
`ca3027a13c67b92e2e181d92a55377ae102fcfd0bd007544ba87cf5dea61c880`.
It freezes **1,941 sources**, including the raw child's complete **1,927-source**
binding. The exclusive future child root is
`go2_commitment_contact_anchored_raw_prefix_v1_attempt_001`.

The fixed saved result is
`f20100955ca2e4bb39c91a9375cb270ff48731d5e721aacf8d5b314821381808`:
observations 0–3, one model forecast, first changed command at 3 from left turn
`[0,0,0.45]` to forward `[0.2,0,0]`. The fresh raw replay must independently
reconstruct that boundary using original and candidate controllers and two
fresh models with state
`755c074325af96d53649aba4927937113d8fab561ea05341a2f3c328598b2bb5`.

The raw runner retains the earlier worker-admission calculation with only the
fixed case index changed from first JEPA to second supervised. It requires the
completed original worker, all original artifact/audit/startup/readout identities,
unchanged sources and full original input verification before and after replay.
It compares all four complete original/candidate decisions and public packets,
full retained contact/map state, model input history and executed residual state.
Every pending forecast is independently checked against the requested action,
raw first-step prediction and current observed pose. The changed request's
pending record is not mistaken for an observed outcome. Observation 4 is never
consumed.

Validation:

- Session 77669: initial 18 raw runner/admission tests passed in 2.48 seconds.
- Session 51062: final 32 raw-runner and waiter tests passed in 2.64 seconds,
  including state corruption, both controllers' input mutation, wrong pending
  action/pose predictions, shared model instances, missing worker terminal,
  changed owners, parent loss, timeout and incomplete result admission.
- Session 70556: raw source-only preflight passed 1,927 source bindings.
- Session 83105: combined preflight verified 1,941 waiter bindings, every child
  binding contained unchanged, exclusive roots absent and exact original worker
  and parent alive. Available resources were 74,136,711,168 RAM bytes,
  644,733,513,728 free artifact bytes, 16 physical CPUs and 5.3% CPU busy.
- Session 6422: waiter launched and emitted
  `WAITING_FOR_ORIGINAL_ADAPTER_SUPERVISED_WORKER`. At the live process check,
  no result/failure or child output existed.

Frozen new files:

| File | SHA-256 |
| --- | --- |
| scripts/replay_go2_commitment_contact_anchored_prefix_v1.py | 69785f558c98f19f72dcd41aff78734756f40d256ffee43c5cd06a4145230403 |
| scripts/await_go2_commitment_contact_anchored_raw_prefix_v1.py | daa27fbf114633645ba0ee0c1a1aadf9b8b2297cece2142d418eb9688502bbf3 |
| lewm/tests/test_commitment_contact_anchored_raw_runner_development.py | 547269be2e8532af233ea037449653bfce6c1b95456b5f789e2ebf2438627a44 |
| lewm/tests/test_commitment_contact_anchored_raw_wait_development.py | fd7f428cacd18911341418fc03c705be93c0f57c9d03eead09e5992518f04115 |
| docs/go2_commitment_contact_anchored_raw_prefix_v1_2026-09-10.md | 8d0435a24e1927630592c8655ecf177ecfb510db84697e9058c4eafa76a420fc |
| docs/go2_commitment_contact_anchored_raw_prefix_wait_v1_2026-09-10.md | 730cc9c185d7eb3d2d43535fc248142fdc636a0d773b81fac21a2cc41e36efb5 |

Launch command:

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONPATH=.:lewm_genesis:lewm_worlds OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .generated/venvs/genesis_rocm_0_4_6_v1/bin/python scripts/await_go2_commitment_contact_anchored_raw_prefix_v1.py
```

The child starts once, only after the original worker ends with complete audit
evidence. It may perform lengthy input hashing before creating its output root;
that is not a stalled or failed attempt while the recorded process is live.
The waiter preserves stdout and result/failure evidence, with no automatic
retry or source changes. Existing batch/frontier/hold native ordering remains
unchanged. This scheduling is not a completed replay, navigation result,
independent-layout comparison, real-time qualification or hardware action.
