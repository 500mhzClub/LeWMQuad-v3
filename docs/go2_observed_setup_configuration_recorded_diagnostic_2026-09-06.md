# Recorded configuration-query diagnostic (no actuation)

Use only the bound 15 RGB-D/body/gyro observations from the completed startup
turn. Rebuild the continuous owner from its original epoch, reproduce the 12
startup decisions and consume the three recorded tail frames. Preserve the
physical trial's result, protocol and all artifacts.

At its final 2.9-s state, query five explicitly supplied rigid translations of
the measured final posture: 0, 0.25, 0.50, 0.75 and 1.00 m along the current
body's forward axis, with unchanged relative orientation and measured joints.
These are geometric configurations, NOT learned dynamics, commanded motion,
reachable postures or sampled certificates for the space between them. Their
local configuration error is zero by construction; inherited pose proxies,
4-cm geometry padding, historical transport and plane/range allowances remain.
Check the existing setup condition through 3.3 s without extending its 3.5-s
expiry. Report unknown residuals, whole-query obstacle vetoes and physical
floor-relation diagnostics separately. Ground support/action permission stays
false even if non-floor configuration clearance is positive.

Check compiled/reference equivalence on every configuration and verify source,
input, native and artifact identities before/after. Report source fingerprints,
per-shape outcomes and observation hashes. This diagnostic neither reruns the
physical trial nor rescores its scientific outcome. Positive results only inform
the next action-conditioned motion/support interface and fresh complete mission.
