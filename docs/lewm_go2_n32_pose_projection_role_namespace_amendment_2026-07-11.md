# Go2 N32 pose-audit role-namespace amendment

Date: 2026-07-11

Status: frozen after the failed-closed first attempt, before corrected runner
output, and superseding the unexecuted train-source filtering proposal.

## Role correction

The current physical-evidence dataset assigns the authoritative roles used for
training, selection, calibration, and untouched G2. Every row in the immutable
pose-audit fit panel has `physical_dataset_role=train` (stored as
`dataset_role=train`).

The V04 summaries also preserve a field named `split` from an older rollout
experiment. Those legacy values are source-lineage metadata, not current
physical-dataset roles. Five of the 20 fit scenes retain legacy values `val`,
`test_id`, or `test_hard`, even though their rendered frames and physical labels
are already members of the current physical training role.

The original audit binding used the unqualified phrase `train-role` and the
first runner incorrectly applied it to both namespaces. It failed closed before
any pose statistic or result was produced. Treating the legacy split as an
access boundary would now discard 76 current-training frames, unbalance family
counts, and bias the registered rough-versus-nonrough comparison. That is not
authorized.

For this audit and all current-architecture claims:

- `physical_dataset_role` exclusively governs role access;
- `legacy_source_split` is recorded provenance only and cannot admit, exclude,
  rank, calibrate, or select a row;
- the exact original 160 transitions / 320 frame records remain in scope;
- physical selection, calibration, untouched G2, non-train payloads, images,
  labels, checkpoints, model outputs, and sealed data remain forbidden; and
- the five legacy non-train source scenes are retired from any future claim
  that their old rollout split was untouched, because the current physical
  training role already uses them.

## Frozen legacy provenance

The corrected runner must require the exact legacy split below for each already
frozen summary path/hash. A mismatch fails closed; the value is reported but
does not change scope.

```text
scene_074f19f0608afca2/summary.json  train      7a5d3b1e6ff5a8acb914ae5226326084c2b951517c110ffc19d7a99945fe0413
scene_142dbd9b0428f16f/summary.json  test_hard  995e192cc1830f32bd2dc6d358da91f5bdaec48bd585ac2dadecc45517cbd2b0
scene_4931dab75d2ceee8/summary.json  train      7800d0d6a14ea54b9970d1dac36472446cd525af8c893736ebe1c4b4bf57cc23
scene_49db95fc9ed0ce8f/summary.json  train      80a035ceecf56f2c668fed3ab1dbabeeca181cb2886fedafa7116ec26bc0566d
scene_4af4d0549179a705/summary.json  train      bcb3866fe141c0c629368eefee8e228630ca8f3b30e1c2810b34e68fd61347b4
scene_7239d51aced24ee3/summary.json  test_id    5c6785479b9a302fcffb1d7532e450af10d2e2625a030eff872edf22b23aef6f
scene_7f390beda8f5070f/summary.json  train      2dc1f874130cb733be4f28eccae3359aac7bdc4e2947718391182ad651d027e7
scene_9ff98ead4f1a2e96/summary.json  train      203ffca9205f68dc74e6135718d3fec4bfb55e9c841bf7a4eb49964930309cc0
scene_a81215e4d326a2a2/summary.json  train      7b9c5dff08be0876327f8b625d225e4b1729320f98b9ccb1efcbd1c68cc2e3c1
scene_b1355439db03d8f8/summary.json  val        d21cd06b202422ecce81c009c08b13ab4e92be86bdc93f6571e69ac265f33fa9
scene_b748962d390baeca/summary.json  train      a3a90172486dc08f3e7a1728da71e43ae224aefddc22ba32e1de5b4fa6ab7f38
scene_b75bb34744434970/summary.json  test_id    64bcf8f57c55cb3456f6dd04be23bbdc417865b2ee8dbad914b5eaa387d61b6b
scene_bc5a05ec9fce8d9c/summary.json  val        41377a7619560162b7fd4453ca302321d2f5f22aee1a8c7397ff32626bbb1a92
scene_c60650f53aaae4a6/summary.json  train      be319a4b1a6e456367c3a6b4d9eee5059380ef83ebe720416b7f292a959c2d6e
scene_cfcadb2bd44cce85/summary.json  train      fa5a9049889a10700cd678fea78ecfb6f91545403ebfdfd304d1dc59a4b6d40a
scene_d8b06cdfb1f739ed/summary.json  train      6f06ee751ec3a26de741bdafcf39cb044e49734cb5a2ab1103ab2834e3edf3c2
scene_ddc88df212918857/summary.json  train      7b1deec174715696d4a3dd653610886e1244edfa993a8c0dc0e91176b728488f
scene_df1c6b34503f2ae1/summary.json  train      deed15024342195754b9022522c048624ab09a1d55e2727f615822d5b6f658e8
scene_e0c2fe611e747d90/summary.json  train      df2fde293612833f00f15a25a8c81c799e15e4674f5ad7f29a0d7ea06e9fd341
scene_ebc33be3e6a87264/summary.json  train      12b5825f4dc2388631190cc80dd42f9cea1bbbbf002f666f12ca53ddde704a35
```

Expected frame-record provenance counts are `train=244`, `test_hard=14`,
`test_id=32`, and `val=30`, totaling 320. Each family retains exactly 64 frame
records.

## Failed-attempt access

The first attempt hashed its two source files and three governing inputs,
parsed the fit-only panel, fully verified the first summary, then parsed the
second summary and stopped at the namespace check. It scanned zero source
`frames.jsonl` files and wrote no result. It opened no RGB, label shard,
checkpoint, model output, G2 payload, or sealed payload.

## Supersession and authorization

This amendment supersedes
`docs/lewm_go2_n32_pose_projection_train_source_scope_amendment_2026-07-11.md`
before that 15-scene proposal was implemented or executed. The corrected runner
must bind and rehash the original audit binding, fit-panel amendment,
superseded-scope amendment, and this role-namespace amendment. Its command
authorization is the exact SHA-256 of this newest amendment.

All geometry, 320-frame scope, five-family balance, float64 summaries, quantile
method, ordering inequalities, output path, and zero model/label/image access
rules from the original audit remain unchanged. The audit still cannot pass
N32, G2, or a runtime gate.
