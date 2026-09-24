# One CPU replay after the original supervised worker audit

Waiter `scripts/await_go2_commitment_contact_anchored_raw_prefix_v1.py` owns
exclusive output `go2_commitment_contact_anchored_raw_prefix_wait_v1_attempt_001`.
It waits for the exact existing full-supervised expanded-model worker:
PID 2672443, creation epoch 1789037034.85, parent PID 2659758. The complete
recorded multiprocessing command, original batch-parent identity, boot identity
and batch launch `97c307ce8d2178494e7484f14b3b0cdcd8d7d0ceaab2f94d94d59da44e39b17a`
are checked. This worker was verified live in its audit at preparation time.
Collection completion alone is insufficient.

Poll every 30 seconds, up to 48 hours. A quiet log, slow audit, missing terminal
while the owner is live, or observation timeout does not permit a restart.
Once the exact worker ends, require its bound terminal and completed original
case audit, then pass that exact terminal SHA to the separately frozen
`replay_go2_commitment_contact_anchored_prefix_v1.py`. Parent loss while the
worker remains live, owner replacement, missing/incomplete terminal evidence,
timeout or child failure produces a retained failure without retry.

Freeze both waiter sources and the child's complete shared seed list, including
the completed saved-result and verification witnesses. The child's independently
discovered source bindings must be contained unchanged in the waiter binding.
Recheck 48 GiB available RAM and 41 GiB free artifact space before child launch;
the child repeats these resource gates after complete input admission.

Launch one CPU child and observe that same process until it exits. The child
owns full original input verification before and after the four raw observations.
After a successful exit, authenticate its result, source/output/launch identities,
original worker terminal, fixed model, all four observations and one forecast,
the changed forward request at observation 3, complete retained observed-state
checks, selected pending forecasts, and no post-intervention observation.
The waiter does not independently rerun neural inference or the child's full
native-input verifier. It preserves stdout and completion receipts.

No native scene or model training is launched by this waiter or its child.
The original six-case batch may continue to later models, and the existing
frontier/hold native waiters keep their ordering. A completed CPU replay grants
no physical navigation, independent-layout, real-time or hardware success.
Any subsequent native pilot needs separate prospective preparation and complete
physical-prefix and raw outcome evidence.
