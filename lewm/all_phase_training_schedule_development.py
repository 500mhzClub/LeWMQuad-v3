"""Matched original update budget with complete expanded-context coverage.

Each original training trial keeps its original total draw weight. Within a
trial, shuffled cycles cover every available expanded context before repeats.
Only the already fixed availability mask controls inclusion; target values and
prediction errors are not consulted. No transfer context enters a batch.
"""
import hashlib
import json
from collections import Counter
import numpy as np
from lewm.all_phase_training_view_development import AllPhaseTrainingView

SEEDS = (2026091001, 2026091401, 2026091402)
UPDATES = 1200
BATCH_SIZE = 6


def schedule(view, *, seed, updates=UPDATES, batch_size=BATCH_SIZE):
    if (not isinstance(view, AllPhaseTrainingView) or type(seed) is not int or seed not in SEEDS
            or type(updates) is not int or updates != UPDATES
            or type(batch_size) is not int or batch_size != BATCH_SIZE):
        raise ValueError('complete study and original three seeds/1200 updates/six samples required')
    source_batches = {}; draw_counts = {}
    for source, expected_trials, draws, seed_offset in (
            ('family', 48, 75, 0), ('switch', 72, 50, 500_000_000)):
        trials = sorted({r['trial'] for r in view.rows if r['source'] == source and r['data_role'] == 'train'})
        if len(trials) != expected_trials: raise ValueError('complete fixed original trial weighting required')
        rng = np.random.default_rng(seed+seed_offset); pool = []
        for trial in trials:
            ids = sorted((i for i in view.indices('train', source=source) if view.rows[i]['trial'] == trial),
                key=lambda i: view.rows[i]['offset_ticks'])
            if not 1 <= len(ids) <= 40:
                raise ValueError('each original trial requires one to forty available contexts')
            selected = []
            while len(selected) < draws:
                selected.extend(int(i) for i in rng.permutation(ids))
            selected = selected[:draws]
            if set(selected) != set(ids): raise ValueError('every available context must be drawn')
            pool.extend(selected); draw_counts[source+'/'+trial] = len(selected)
        if len(pool) != 3600: raise ValueError('exact original source draw budget required')
        pool = [pool[int(i)] for i in rng.permutation(len(pool))]
        source_batches[source] = [pool[i:i+6] for i in range(0, 3600, 6)]
    rng = np.random.default_rng(seed+1_000_000_000); batches = []
    for i in range(600):
        pair = [source_batches['family'][i], source_batches['switch'][i]]
        batches.extend(pair[int(j)] for j in rng.permutation(2))
    counts = Counter(i for batch in batches for i in batch)
    if set(counts) != set(view.indices('train')):
        raise ValueError('exact complete available training population and no transfer draws required')
    record = dict(role='train', seed=seed, updates=UPDATES, batch_size=BATCH_SIZE, batches=batches,
        sample_ids=[r['sample_id'] for r in view.rows], source_batches=dict(family=600, switch=600),
        trial_draw_counts=draw_counts, context_draw_counts={str(i): counts[i] for i in sorted(counts)},
        available_training_contexts=len(counts), total_draws=7200,
        weighting='75 draws per original family training trial;50 per original switch training trial',
        within_trial_sampling='shuffled available-context cycles, truncated to fixed original trial weight',
        availability_mask_fixed=True, target_values_used_for_sampling=False,
        prediction_errors_used_for_sampling=False, geometry_transfer_draws=0)
    sha = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return record | dict(schedule_sha256=sha)
