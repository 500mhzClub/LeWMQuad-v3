# Go2 N32 pose-audit train-source scope amendment

Date: 2026-07-11

Status: frozen after the first audit attempt failed closed and before the
corrected runner or any pose-audit result.

## Failed-closed attempt

The first authorized attempt produced no result. It stopped while validating
the second frozen V04 scene summary because the summary's source rollout split
was `test_hard`, not `train`. The physical-head panel row itself correctly had
`dataset_role=train`; these are distinct role systems that the original audit
binding incorrectly assumed were identical.

Before failing, the runner hashed its two local source files, the original
binding, the fit-panel amendment, and the fit-only panel. It parsed the fit-only
panel, hashed/parsed/rehash-verified
`scene_074f19f0608afca2/summary.json`, and hashed/parsed
`scene_142dbd9b0428f16f/summary.json`. Source `frames.jsonl` scanning occurs only
after every summary validates, so it opened zero source-frame files. It opened
zero RGB, label shard, checkpoint, model output, G2 payload, sealed payload, and
result files. The immutable result path remained absent.

The original binding forbids non-train source roles. Relaxing that rule after
this discovery is not authorized.

## Corrected scope

The corrected audit still consumes the immutable fit-only artifact:

- file SHA-256:
  `77d84e242d75b81fd2b96f086e9cf5df72f0a907e1fe7ce24fc48bbc5d514037`;
- content SHA-256:
  `8e44dd0238077120e97fd06b4550d6504627066c7e8ddfdfbd138fd7504ee7a8`.

It deterministically retains a transition only when both committed frame paths
resolve to the same scene directory in this exact train-source summary
allowlist:

```text
scene_074f19f0608afca2/summary.json  7a5d3b1e6ff5a8acb914ae5226326084c2b951517c110ffc19d7a99945fe0413
scene_4931dab75d2ceee8/summary.json  7800d0d6a14ea54b9970d1dac36472446cd525af8c893736ebe1c4b4bf57cc23
scene_49db95fc9ed0ce8f/summary.json  80a035ceecf56f2c668fed3ab1dbabeeca181cb2886fedafa7116ec26bc0566d
scene_4af4d0549179a705/summary.json  bcb3866fe141c0c629368eefee8e228630ca8f3b30e1c2810b34e68fd61347b4
scene_7f390beda8f5070f/summary.json  2dc1f874130cb733be4f28eccae3359aac7bdc4e2947718391182ad651d027e7
scene_9ff98ead4f1a2e96/summary.json  203ffca9205f68dc74e6135718d3fec4bfb55e9c841bf7a4eb49964930309cc0
scene_a81215e4d326a2a2/summary.json  7b9c5dff08be0876327f8b625d225e4b1729320f98b9ccb1efcbd1c68cc2e3c1
scene_b748962d390baeca/summary.json  a3a90172486dc08f3e7a1728da71e43ae224aefddc22ba32e1de5b4fa6ab7f38
scene_c60650f53aaae4a6/summary.json  be319a4b1a6e456367c3a6b4d9eee5059380ef83ebe720416b7f292a959c2d6e
scene_cfcadb2bd44cce85/summary.json  fa5a9049889a10700cd678fea78ecfb6f91545403ebfdfd304d1dc59a4b6d40a
scene_d8b06cdfb1f739ed/summary.json  6f06ee751ec3a26de741bdafcf39cb044e49734cb5a2ab1103ab2834e3edf3c2
scene_ddc88df212918857/summary.json  7b1deec174715696d4a3dd653610886e1244edfa993a8c0dc0e91176b728488f
scene_df1c6b34503f2ae1/summary.json  deed15024342195754b9022522c048624ab09a1d55e2727f615822d5b6f658e8
scene_e0c2fe611e747d90/summary.json  df2fde293612833f00f15a25a8c81c799e15e4674f5ad7f29a0d7ea06e9fd341
scene_ebc33be3e6a87264/summary.json  12b5825f4dc2388631190cc80dd42f9cea1bbbbf002f666f12ca53ddde704a35
```

The resulting scope is exactly 122 transitions and 244 frame records:

| Family | Transitions | Frames |
| --- | ---: | ---: |
| open obstacle field | 32 | 64 |
| rough local dynamics | 18 | 36 |
| small enclosed maze | 23 | 46 |
| medium enclosed maze | 17 | 34 |
| large enclosed maze | 32 | 64 |

The runner must filter from committed fit-row path metadata before resolving,
hashing, or parsing any scene summary. It must prove that all 15 allowlisted
summaries declare `split=train`, and it must never resolve or open these five
excluded summary paths or their source files:

```text
scene_142dbd9b0428f16f/summary.json
scene_7239d51aced24ee3/summary.json
scene_b1355439db03d8f8/summary.json
scene_b75bb34744434970/summary.json
scene_bc5a05ec9fce8d9c/summary.json
```

The access ledger must distinguish the original 320 fit-frame commitments from
the 244 selected train-source records and reconcile 76 excluded records with
zero excluded-summary or excluded-source byte opens.

## Unchanged decision

All geometry, float64 summaries, quantile method, and ordering thresholds from
the original audit binding remain unchanged. In particular, the next
intervention is dynamic pose only when rough-local-dynamics median per-frame p50
displacement is `>=0.5` token and exceeds the pooled selected non-rough median
by `>=0.25` token. Unequal selected family counts are reported explicitly and
the non-rough estimand remains the preregistered pooled median.

The corrected runner must bind and rehash the original audit binding, fit-panel
amendment, and this amendment. Its command authorization must be the exact file
SHA-256 of this amendment, not the original binding hash. It uses the same
exclusive result path because no prior result exists.

This audit remains metadata-only. It cannot pass N32, G2, or a runtime gate.
