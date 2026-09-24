# Swept-progress survival labels V1 preflight result

Date: 2026-07-28

Status: **PASS — one capped joint-JEPA development probe may proceed.**

- Label-builder source commit at launch: `8f654bd`
- Terminal repository HEAD after the independently reviewed neural mask-frame correction: `ab24725`
- Label manifest content SHA-256: `6e0ea572612cdf94cb6dd91dffb90e50c828053617f69b42307161c958700c03`
- Label manifest file SHA-256: `edc0df8c796f97d3f91c8c3796e9795a4355dceac79770b91de382132fe8e1d3`
- Population: 5,172 development states and 46,548 nine-action rows.
- Informative states: train 3,546 / 4,262; probability calibration 337 / 415; checkpoint selection 399 / 495.
- Frozen 16,000-presentation schedule: 13,310 informative presentations.
- Each non-HOLD action participates in an unequal-prefix comparison on 13,310 scheduled presentations.
- Selection informative states by family: large enclosed 64, local motifs 51, loop alias 61, medium enclosed 64, open obstacle 26, rough dynamics 22, small enclosed 47, visual stress 64.
- Output files: train `1ed44b637e91263752502a1c55d26034f599d473cb1018fce8be196df785b7f8`; calibration `0937a10cef5e1f4db23332c5493bb757317b22a6520da718faab70eb24952088`; selection `bae8973396b3536fb3a3465da69a9bcb49ff49ad1c736f4c9327b55b1308b7cc`.
- Access: the exact 88 development scenes were scanned once. RGB, model, checkpoint, GPU, training, runtime output, navigation, G2, held-out, sealed, and production open counts were all zero.
- Authority: this PASS authorizes only source completion and one fresh development attempt capped at 1,000 updates / 16,000 presentations. It does not authorize G2, held-out, sealed, navigation, promotion, or a JEPA treatment-effect claim.

The target-coverage question is closed positively. The next decision is neural: whether the jointly trained RGB JEPA can use this target better than persistence, shuffled-action, wrong-RGB, and action-prior controls.
