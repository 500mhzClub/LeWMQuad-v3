#!/usr/bin/env bash
set -euo pipefail

task_root=/home/andrewknowles/Workspace/LeWMQuad-v3
cd "$task_root"
/usr/bin/sha256sum --check <<'BINDINGS'
bc405303fb8f7d227c71bf7e965d2262ca6ad8f8e817e78512427bbacf8eaa77  docs/go2_gsd_cache_retirement_proposal_2026-09-08.json
c55dfbe858b7d6e17107189a38c6bc3187afc9c9d2b72b05959a52daa705ca5e  docs/go2_gsd_cache_retirement_authorization_2026-09-08.json
c9ecbfae608bafd55f28c78bcf8be3b7baa5156526185cc1ab55cdc07d5306d1  scripts/retire_go2_reviewed_geometry_cache_privileged_v1.py
BINDINGS

exec sudo /usr/bin/env \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
    PYTHONPATH="$task_root:$task_root/lewm_genesis:$task_root/lewm_worlds" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$task_root/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python" \
    "$task_root/scripts/retire_go2_reviewed_geometry_cache_privileged_v1.py" \
    --proposal-sha256 bc405303fb8f7d227c71bf7e965d2262ca6ad8f8e817e78512427bbacf8eaa77 \
    --authorization-file "$task_root/docs/go2_gsd_cache_retirement_authorization_2026-09-08.json" \
    --authorization-sha256 c55dfbe858b7d6e17107189a38c6bc3187afc9c9d2b72b05959a52daa705ca5e
