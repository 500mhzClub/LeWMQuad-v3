# Scoped verification digest preparation

Static direct-call analysis of the height-prefix verifier found 26 unique
functions and 1,227 expanded call paths, including 263 verify_artifacts paths,
399 digest paths and 44 paths to common launch/native checks. These are static
path multiplicities, not observed runtime counts or physical disk reads.
This supports testing bounded digest reuse; it does not justify removing checks.

Implemented an isolated scope retaining every original verification condition.
Each distinct file first executes the original digest; guarded repeats reuse
that value only within the call. Every cached file receives a fresh independent
streaming SHA-256 and identity check before success, followed by a final
population metadata check. Original function code/closures/defaults remain
unchanged in isolated namespaces; no imported globals or frozen files mutate.
No cache survives success or failure. See the benchmark protocol for limits.

Initial synthetic run had two failures because verifier functions in the test
module were not traversed. The implementation now admits the explicit root's
module and discovers referenced globals in nested code objects as well as the
outer body. A comprehension regression test covers this. The final helper suite
passes 20 tests in 0.14 s (829959 output, exit 0); the paired-runner suite passes
5 in 2.19 s (35467 exit 0). Tests cover content/identity changes, inode replacement,
restored modification time, modes, symlinks, protected paths, mutation during
hashing and later population hashing, original digest execution, independent
final hashing, input preservation, original failures and retained stages.

Fresh hardware 92285 exit 0: 16 physical/32 logical/all affinity, CPU 6.5%; RAM
74,854,051,840 bytes available, artifact free 75,698,769,920 bytes, workspace
free 21,358,641,152 bytes. Both GPUs idle; card 1 VRAM total 34,208,743,424 bytes,
used 1,398,722,560. Hold/height replay RSS 5,971,136,512 / 2,764,222,464 bytes.
Their two 8 GiB allowances plus this 8 GiB CPU benchmark fit. No native scene.

Frozen sources:

| Path | SHA-256 |
| --- | --- |
| scripts/scoped_verification_digest_development.py | 4c7d078cee2e4d9cc6487173aa9d404ce115c456070faedc51e0aeb87ac9ffb8 |
| scripts/benchmark_go2_scoped_verification_digest_v1.py | 11ec248a7229afd67eb7c1dd3d6e114f941f6091edd44336952dab48b0200846 |
| lewm/tests/test_scoped_verification_digest_development.py | eab40b462fe9c626fe8e7e882ca66302a5dc991ba0188746b67bff0bdb4d6ce9 |
| lewm/tests/test_scoped_verification_digest_benchmark_development.py | c59c90b39c7521ce032ca1bbef197fa5e4964ab48c67e1c53e9bb4ef256eb752 |
| docs/go2_scoped_verification_digest_benchmark_v1_2026-09-09.md | 5c051d7d3776a3585869fe7c76f673fb60de7792974406cf6d9adb22b4f3c712 |

Run scripts/benchmark_go2_scoped_verification_digest_v1.py in the existing
Genesis interpreter with PYTHONDONTWRITEBYTECODE=1, PYTHONHASHSEED=0,
PYTHONPATH=.:lewm_genesis:lewm_worlds and one OMP/MKL/OpenBLAS thread. Exclusive
output go2_scoped_verification_digest_benchmark_v1_attempt_001. This fixed pair
does not modify existing or prepared launchers and is not a controlled timing
study. Preserve its outcome before deciding whether a separately reviewed
future launcher may use the helper.
