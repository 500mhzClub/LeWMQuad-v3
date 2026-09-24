# N32 fit-capacity keyed-oracle result

Date: 2026-07-12  
Status: development-only diagnostic passed; no model/generalization license

The perfect image-identity lookup passed the unchanged complete N32 fit gate,
including aggregate, all five families, distance bins, and both wrong-image
controls. This proves that the labels, controls, metric implementation, and
numeric thresholds are jointly attainable on the registered 320-frame fit
panel.

- result file SHA-256:
  `c77573edf436a7ddc07d97281bc11c79469c42a5cb00c51d8ce3a0a488ae6b4c`
- canonical content SHA-256:
  `a6c9601b229c57e97e551742b0829c2d9c3d9d1cd47b472e44bba30c674809ea`
- hierarchical NLL: `4.540256345860287e-11`
- UNKNOWN/KNOWN balanced accuracy: `1.0`
- FREE/OCCUPIED balanced accuracy: `1.0`
- UNKNOWN/FREE/OCCUPIED recall: `1.0 / 1.0 / 1.0`
- global wrong-image NLL delta: `7.341745553718303`
- same-scene wrong-view NLL delta: `5.868415490091175`
- all-family gates: `5/5`

The diagnostic opened the 20 fit label shards and no image payload. It opened
no non-fit, checkpoint-selection, calibration, G2, held-out, or sealed
payload. It has no learned parameters and cannot support a model or
generalization claim.

Combined with the two-seed dynamic-Cartesian result, this isolates the failure
to the learned visual representation/decoder/optimization path rather than an
impossible fit contract. The next comparison must retain the same labels and
gate while separating the existing soft Cartesian lift from an explicit
camera-ray depth/first-surface decoder and a stabilized optimizer control.
