# Combined packed insertion and single-pass query timing

Completed result: `85bcc450c6286c624c976b49cbc468d366275c87a80deec175439a967211def9`.
Launch: `1976e3b5d93cdef6c8fa2c379416dfa1ebf6e66d7de2708dc5b458a4e84602a4`.
Both compressed decision streams:
`ae76078064eed0096398866bb5c9ee7ffc00cd418e00ee3991dc359ae6d70144`.
Root: `go2_single_pass_controller_pair_v1_attempt_001` in the development
navigation artifact store. Wall time 404.301770772 s; 1557 bound sources.

The two independent controller/model instances processed the same first 256
observations, alternating which ran first each frame. All complete decisions
match the original episode. Model state and input arrays remain unchanged;
final source/input checks and compressed-stream identity pass.

| Metric | Original median | Candidate median |
| --- | ---: | ---: |
| Controller observe | 707.718 ms | 604.847 ms |
| Receipt writing | 17.245 ms | 17.299 ms |
| Controller plus receipt | 726.093 ms | 624.690 ms |
| Controller plus receipt process CPU | 726.039 ms | 624.090 ms |

Median per-frame paired ratios are 1.183122 for observe and 1.177638 including
receipt. **All 256 frames in both arms exceed the 100 ms deadline.** Maximum
controller-plus-receipt costs are 959.849 ms original and 862.979 ms candidate.

This measures combined packed-owned insertion and single-pass queries against
the original controller. It does not isolate query changes against the earlier
packed-only candidate. The earlier packed-only and present results come from
separate shared-machine runs and do not establish an incremental speed gain.
Sensing, decoding, private input copies, comparison and physics are outside
timing. Only an early prefix was measured; no whole-trajectory or native-loop
speedup is established. The native settling collector has not adopted either
optimization. Full-goal timing and navigation requirements remain unfinished.
