"""Same 7,200-draw budget, adding pulse contexts without losing original ones."""
from collections import Counter
import hashlib
import json
import numpy as np
from scripts import pre_switch_training_data_development as old
from lewm.eligible_floor_registration_development import bind

BASE=old.BASE
PULSE=BASE/'go2_short_pulse_learning_v1_attempt_001'
OUTPUT=BASE/'go2_short_pulse_training_schedule_v1_attempt_001'
ROOTS=old.ROOTS|dict(short_pulse=PULSE)
prepare=bind(old.prepare,ROOTS=ROOTS)


def load_training_rows():
    pulse=json.loads((PULSE/'windows.json').read_text())
    return old.load_training_rows()+[r for r in pulse if r['available'] and r['data_role']=='train']


def main():
    if OUTPUT.exists():raise ValueError('preserve pulse schedule')
    terminal=json.loads((PULSE/'result.json').read_text())
    raw=(PULSE/'windows.json').read_bytes()
    if terminal['status']!='COMPLETE' or hashlib.sha256(raw).hexdigest()!=terminal['windows_sha256']:
        raise ValueError('completed collection required')
    previous=BASE/'go2_pre_switch_training_schedule_v1_attempt_001/schedule.json'
    schedule=json.loads(previous.read_text());rows=load_training_rows();by_id={r['sample_id']:r for r in rows}
    added=[r['sample_id'] for r in rows if r['source']=='short_pulse']
    if len(added)!=180 or len(rows)!=4694:raise ValueError('fixed available pulse population required')
    batches=[list(b) for b in schedule['batches']];counts=Counter(i for b in batches for i in b)
    rng=np.random.default_rng(2026091602)
    positions=[(i,j) for i,b in enumerate(batches) for j in range(len(b))]
    positions=[positions[i] for i in rng.permutation(len(positions))]
    replacements=[]
    for i,j in positions:
        identifier=batches[i][j]
        if counts[identifier]>1:
            replacements.append((i,j));counts[identifier]-=1
        if len(replacements)==900:break
    if len(replacements)!=900:raise ValueError('insufficient repeated draws; do not omit old contexts')
    pulse_draws=[str(i) for i in rng.permutation(added*5)]
    for (i,j),identifier in zip(replacements,pulse_draws,strict=True):batches[i][j]=identifier
    counts=Counter(i for b in batches for i in b)
    if set(counts)!=set(by_id) or sum(counts.values())!=7200:raise ValueError('same budget and all contexts required')
    record=schedule|dict(batches=batches,available_contexts=len(rows),added_contexts=180,
        context_draw_counts=dict(counts),trial_draw_counts=dict(Counter({
            key:sum(counts[r['sample_id']] for r in rows if r['source']+'/'+r['trial']==key)
            for key in {r['source']+'/'+r['trial'] for r in rows}})),
        predecessor_schedule_sha256=hashlib.sha256(previous.read_bytes()).hexdigest(),
        input_sha256=schedule['input_sha256']|{str((PULSE/'windows.json').relative_to(BASE)):hashlib.sha256(raw).hexdigest()},
        replaced_repeated_original_draws=900,pulse_draws=900,every_original_context_retained=True,
        unchanged_family_batches=sum(b==old_b for b,old_b in zip(batches,schedule['batches'])
            if all(by_id[i]['source']=='family' for i in old_b)),
        original_per_trial_draw_weights_preserved=False,
        sampling_uses_targets_or_prediction_errors=False,geometry_transfer_draws=0)
    OUTPUT.mkdir();(OUTPUT/'schedule.json').write_text(json.dumps(record,indent=2)+'\n')
    print('SHORT_PULSE_SCHEDULE',len(rows),'contexts, 7200 draws, 900 pulse draws; all old contexts retained')


if __name__=='__main__':main()
