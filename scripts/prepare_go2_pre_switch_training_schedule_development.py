"""Keep the original draw budget while including recovered training contexts."""
from collections import Counter
import hashlib
import json
import numpy as np
from scripts.read_go2_training_execution_coverage_development import BASE

OUTPUT = BASE/'go2_pre_switch_training_schedule_v1_attempt_001'
SEED = 2026091001


def main():
    if OUTPUT.exists():
        raise ValueError('preserve fixed schedule')
    names = ('go2_all_phase_training_targets_v1_attempt_001/windows.json',
             'go2_pre_switch_training_targets_v1_attempt_001/windows.json',
             'go2_all_phase_matched_fits_v1_attempt_001/training_schedules.json')
    data = [(BASE/n).read_bytes() for n in names]
    old, added, schedules = map(json.loads, data)
    source = schedules[str(SEED)]
    rows = [r for r in old+added if r['available']]
    by_id = {r['sample_id']:r for r in rows}
    if len(rows) != 4514 or len(by_id) != len(rows) or any(r['data_role'] != 'train' for r in rows):
        raise ValueError('same original plus recovered training contexts only')
    rng = np.random.default_rng(SEED+500_000_000); pool = []
    trials = sorted({r['trial'] for r in rows if r['source'] == 'switch'})
    for trial in trials:
        ids = [r['sample_id'] for r in sorted((r for r in rows if r['trial'] == trial),
                                             key=lambda r:r['decision_ns'])]
        selected = []
        while len(selected) < 50:
            selected.extend(str(i) for i in rng.permutation(ids))
        selected = selected[:50]
        if set(selected) != set(ids):
            raise ValueError('every available switch context must be covered')
        pool.extend(selected)
    pool = [pool[int(i)] for i in rng.permutation(len(pool))]
    batches = []; offset = 0; unchanged_family = 0
    for batch in source['batches']:
        ids = [source['sample_ids'][i] for i in batch]
        sources = {by_id[i]['source'] for i in ids}
        if sources == {'family'}:
            batches.append(ids); unchanged_family += 1
        elif sources == {'switch'}:
            batches.append(pool[offset:offset+6]); offset += 6
        else:
            raise ValueError('original source-homogeneous batch required')
    counts = Counter(i for batch in batches for i in batch)
    trial_counts = Counter()
    for sample, count in counts.items():
        row = by_id[sample]; trial_counts[row['source']+'/'+row['trial']] += count
    if (len(batches) != 1200 or any(len(b) != 6 for b in batches)
            or unchanged_family != 600 or offset != 3600 or set(counts) != set(by_id)
            or dict(trial_counts) != source['trial_draw_counts']):
        raise ValueError('original budget and per-trial weights must stay exact')
    record = dict(seed=SEED, updates=1200, batch_size=6, total_draws=7200,
        available_contexts=len(counts), added_contexts=504, unchanged_family_batches=unchanged_family,
        batches=batches, trial_draw_counts=dict(trial_counts), context_draw_counts=dict(counts),
        models=['full_jepa', 'full_direct', 'full_supervised_rollout'],
        predecessor_schedule_sha256=source['schedule_sha256'],
        input_sha256={n:hashlib.sha256(d).hexdigest() for n,d in zip(names,data)},
        sampling_uses_targets_or_prediction_errors=False, geometry_transfer_draws=0,
        model_training_started=False)
    OUTPUT.mkdir()
    (OUTPUT/'schedule.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('batches','trial_draw_counts','context_draw_counts')}))


if __name__ == '__main__':
    main()
