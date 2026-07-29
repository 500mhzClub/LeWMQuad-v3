# V9 Dense Local-Token Lift Physical Calibration — Preimplementation Amendment

- Status: frozen before adapter/runner completion and before any V9 checkpoint
  or calibration/selection payload access.
- Authority: narrow source-closure correction to preregistration
  `2f561d26f0b6ca154b6f4eab00dba228f8bc8c9e`; it changes no candidate,
  data, calibration, threshold, gate,
  stopping rule, or access scope.
- The preregistration listed the direct raw contract and four loader/constructor
  sources but omitted two transitive benchmark implementations that the frozen
  V4 calibration runner also validates. Preserve the complete V4 source closure
  by additionally binding:
  - `lewm/benchmarks/go2_rgb_jepa_encoder_pretraining_v1.py` SHA-256
    `8c35f0cbafe78185ac74d4412914c177de20f899b0f009a9b9dc7aafdf7695a5`.
  - `lewm/benchmarks/go2_shared_jepa_v5_matched_training_v1.py` SHA-256
    `53a7fac793a1b46764d49e7259fd637ec02b20111927effd01cdcd09682c206a`.
- The V9 runner must validate the union of the original preregistered hashes,
  these two hashes, its own frozen adapter/runner sources, and the V9 model
  source before candidate access.
