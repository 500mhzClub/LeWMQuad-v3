# Actual maze camera provenance recorder verified at startup

Bounded probe58428 CLOSED exit0, root
go2_maze_renderer_witness_probe_v1_attempt_001. Launch
1158e27563615dafa02d3830a11c0bb8124cf05c473801fa50e6a7e6783950c4,
1631 frozen source bindings. Result
5107b9a48c9767149b0f1a217875065f6e98489fe97885626cca9a0ace547f88,
wall63.929525689s.

Three unchanged zero-command warmup observations and900physical samples were
collected in the actual maze/robot camera wrapper. Acquisition stopped at the
declared three-observation boundary, with no physical/controller terminal.
This is a camera integration probe, not a twelfth navigation episode.

Full raw sensor/model/controller/actual-command audits and model-state checks
pass. Every startup physical sample, complete decision, public sensor packet
and raw primary/auxiliary capture matches completed eleventh native result
44710966178a57b31f7da3bec10ad4f21710bcff3701b0a1038750f1ef6d747c.
Physical-prefix SHA8419be1a3143128fcef2a1cf843d6476063177cc52f83316c42fd9b2bc7789fa.

All six context endpoint witnesses match the independently audited acquisition
hashes, clocks, physical sample indices and primary optical poses. Paired
context readbacks are equal; no query failure, extra rendering, physics step,
public packet mutation or pose change was recorded. The after-pair endpoint
follows primary pose restoration; it does not claim that current framebuffer
contents were rendered from the restored pose.

- Witness artifact2e198936811ae22bb7ec0da8cf8693abd9e94f1de51e08d262d01ba8c7a6cfa1.
- Raw audit4ec190ed90904111a6800a5ed35c1194024e7cc36c460c9e3b9c1e94ccff01d7.
- Comparison8cc84fa5aaec74bd7b8c87f6fdd080a5ed18c3eaf46a82d96b63c68383445a20.

The probe validates recorder integration at startup. It does not recover a
historical renderer context, reconstruct each shader draw, prove a raster
error bound or resolve strict visibility failure909. A future episode can bind
this tested recorder explicitly and must audit its own complete acquisitions.
No navigation, real-time or hardware deployment qualification is claimed.
