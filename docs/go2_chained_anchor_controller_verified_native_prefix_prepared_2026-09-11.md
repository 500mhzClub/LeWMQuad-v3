# Controller verification and physical-prefix preparation

The full chained-anchor controller replay completed with result SHA-256
`d68e5e48916ff84d4034c95a2af5357695d277048519a15cf5dbb6a554703231`.
Its original process (PID 2840884, creation time 1789126982.74) ended.
The independent completion check reconstructed every consumed public packet,
every saved comparison, and the entire result report. It checked the original
execution identity, 2270 source bindings, and bound replay and native-worker
artifacts. It did not rerun the policy or floor registration.

Completion evidence:
`docs/go2_chained_anchor_controller_completion_verification_2026-09-11.json`,
SHA-256 `6b0c3298bb4df47aa59affc80d5dcb9236c1e4d9ae5cabd07550800c94d26425`.
The checker exited successfully (session 95543). Its 21 focused tests and the
20 existing controller-comparison tests passed together in 2.25 seconds
(session 91446).

The replay consumed 854 observations. All 853 earlier complete controller
decisions and 850 earlier forecasts matched. At frame 853, the controller
accepted the reacquired anchor and requested the same right turn as the
original controller. The original controller had not failed at that boundary.
No following recorded observation was consumed and no new command was executed.
The assigned no-RGB JEPA model was unchanged in the original replay.
This result supports a prospective experiment, not a claim of recovered
navigation, a round trip, or an independent-layout improvement.

The new helper `scripts/chained_anchor_native_prefix_development.py` requires
43400 identical preintervention physics samples, all 854 identical public
packets, and all complete candidate decisions through the intervention. It
requires completed actual command tapes and 50 boundary-command physics samples.
It explicitly records that the command did not change at the evidence boundary
and that following physical outcomes have not been compared.

All 21 focused physical-prefix tests passed in 2.79 seconds (session 89831).
They reject changed physics, missing samples, incomplete or changed commands,
changed sensor packets, changed forecasts, incomplete populations, and false
navigation claims. The actual completed replay also passed reconstruction with
this new helper (session 99680). Preparation evidence:
`docs/go2_chained_anchor_native_prefix_preparation_2026-09-11.json`, SHA-256
`6acce51f98106224761b7987ccd3a58a2624a1975a501a6c01211b2ab182bc5e`,
binding 2280 source paths. The existing collector and auditor remain unchanged.

During this work, the original tracking waiter (PID 2753911, creation time
1789076672.54) ended with a completed audit. Its result SHA-256 is
`361ad8f517a8cef47384931156adb2341c749fb10d92fa06732ac9c67539fcee`, binding
native result `ce80ef3dffb6a249acfb3ea11dcbab459206ae6a088046b4a6a325d60e0c1bc1`.
It reports zero round trips. The existing extended-budget waiter subsequently
started its original native launcher: PID 2843773, creation time
1789128335.77. Its exact command binds the original batch, frontier, hold,
contact, and newly completed tracking-waiter results. The child was confirmed
running directly through the process table; no physics progress or outcome is
claimed from that launcher status. No process was restarted.

Next work is the source-bound chained-anchor launcher and ordered waiter,
preserving the existing diagnostic order through extended budget, sustained
turn, and contact-plus-flow. A fresh chained-anchor physical outcome is still
missing. Reliable unseen-maze navigation, causal advantages of JEPA/planning/
memory, realistic timing, and bounded real-platform evidence remain unproven.
