# Completed scoped/original verification pair

Benchmark session 20770 exited 0. Result
137867773bfe6c6eb05a125a288012ff6017aa3134f4687d7bccdc7f99c02071,
root go2_scoped_verification_digest_benchmark_v1_attempt_001. All 1,689 source
bindings and all output bindings were independently rechecked after completion
(88182 exit 0). Both stages verified the same completed tracking native inputs
and left the context unchanged.

Scoped stage: 138.461881753 s. It received 1,673,231 digest requests, reused
1,550,401 guarded values, and hashed 122,830 unique files totaling 56,312,123,706
bytes. Every cached file was freshly hashed again at the end, another
56,312,123,706 bytes, with matching content and metadata plus final population
metadata checks. 23 verification functions used isolated namespaces with
unchanged code/closures/defaults/conditions. No imported module globals changed
and no cached value survived the call.

Original stage: 354.217828018 s, completed successfully without scoped reuse.
This is one fixed-order pair with scoped first; the observed difference is not
a controlled speedup estimate. It does establish successful completion of both
paths on this actual frozen input population. Total benchmark wall time after
admission and final direct checks: 493.863733615 s. No model or native scene ran.

Bindings:

- launch.json: a6d3d860c00cc3116a67429b38ef420b298808953331d65d6812a0f760a78022
- scoped_verification.json: 9cb36e5c8bd39504a2add3577f3d43a1ac942f7e4b8ed5e8f3d8dbb823d111b1
- original_verification.json: 0e70e501a4832fe7d5a143b37a9f39dcaf3ac75cbcca80a1958f2396e781190d

The fixed helper and tests remain unchanged. Any future adoption must bind this
completed result and retain the original scope limits and fresh final checks.
No running verifier is modified by this result. It establishes no new navigation
outcome or reduction in controller computation time.
