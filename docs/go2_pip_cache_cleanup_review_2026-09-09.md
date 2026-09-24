# Narrow package-cache cleanup proposal

Read-only inspection found142 ordinary files under
/home/andrewknowles/.cache/pip, occupying1,528,332,288 allocated bytes
(approximately1.423GiB). The exact names and device/inode/size/ownership/link/
timestamp metadata are fixed in docs/go2_pip_cache_cleanup_proposal_2026-09-09.json,
SHA-25695d9acb80c802893f40f7cee236b80eec6f8d4845afe3fc981bb4c28147a6381.
No payload was opened, changed or removed. All142 files are UID1000-owned,
single-linked ordinary files. The walk explicitly rejected protected names,
symlinks and nonordinary entries; none occurred. Preserve all3573 directories
and every unlisted file. Do not use recursive directory deletion.

Accessible process metadata showed no pip/uv/conda/mamba installer and no open
file under this cache.45 processes were inaccessible, so this is not a universal
open-file/dependency proof. The active residual launch's explicit binding and
environment fields contain no literal reference to the cache. That is a narrow
binding check, not a claim about every historical workflow. No installed
environment, model, dataset, simulation cache or scientific evidence is targeted.
The practical cost is needing to download cached packages again when required.

Measured artifact free space78,112,460,800bytes is below the fixed supervised
three-case admission requirement78,383,153,152bytes. This proposal could recover
about1.53GB, giving limited headroom. The preceding tracking simulation can
consume that headroom; this cleanup alone does not guarantee the whole queue
will fit. Do not reduce frozen resource reserves or omit comparison cases.

Prior approval explicitly excludes pip cleanup:
docs/go2_gsd_cache_retirement_authorization_2026-09-08.json contains
"pip_cache_cleanup_included": false and authorizes only8,304 named GSD files.
This proposal is not authorization and requires a new explicit user decision.

If approved, recheck active package tasks and accessible open files, cache-root
and parent identities, and every listed leaf's metadata before the first unlink.
Reject changed files or symlink/protected path components; never broaden scope
to new cache files. Recheck each leaf immediately before deletion and preserve
an exclusive journal of actual removals. Keep directories and all unlisted
files. Report actual recovered capacity and refresh native resource admission.
If approval does not arrive, preserve the cache and continue independent work.
