# Downward-camera return-transition pair diagnosis

Completed V2 result: `4980170b5c02e9ab8de55f331583f7b0070decdc53aa753dbe9466821cf855da`.
Launch: `b8d90d03570043029353d6bac02fd4b89fac827bca26402c2bab1a688c214544`.
Artifact root: `go2_auxiliary_return_pair_diagnosis_v2_attempt_001` in the
development navigation artifact store. Wall time 1.238170434 s; 1548 source
bindings. Input/source final checks pass. V1's guarded-path failure remains
preserved; V2 corrected the attempt-root argument without changing fitting.

On the fixed 18 frames (1853–1870), the original front-camera counts and
feature witnesses reproduce exactly. Detector, matching and registration
thresholds are unchanged. Fixed camera extrinsics convert auxiliary gyro and
fitted poses into the appropriate reference/body frames, including lever arm.
Native robot poses enter only after fitting, for error evaluation.

| Pair population | Front accepted | Downward accepted | Downward maximum translation / rotation error |
| --- | ---: | ---: | --- |
| 17 consecutive pairs | 15 | 17 | 0.419 mm / 0.000681 rad |
| 8 retained-reference pairs ending at 1870 | 0 | 4 | 1.016 mm / 0.002169 rad |

At the original failure transition 1869→1870, front RGB supplies eight lifted
matches and fails. Downward RGB supplies 22 and fits with 0.196 mm translation
error and 0.000378 rad rotation error. Downward frame 1870 has 94 selected
features versus 22 front features. These are observed errors, not uncertainty
bounds. Retained pairs 1866, 1859 and 1853 fail for insufficient matches;
1861 fails consensus pruning. All four failures remain recorded.

This supports testing downward RGB in a causal observer. It does not prove
continuous online tracking: the retained indices came from the primary
observer, and auxiliary retention/bridge state was not run. RGB is an
additional captured but currently unused controller modality; no controller
or frozen native attempt was modified. Next requirements are a distinct public
RGB/depth packet, continuous observer replay, then prospective integration and
native validation. The ninth failed navigation result, strict visibility
failure, zero verified round trips and full-goal limitations remain unchanged.
