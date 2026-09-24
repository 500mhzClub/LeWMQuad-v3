"""Same-budget command-history fit and separate old/pulse transfer scores."""
import hashlib
import json
import numpy as np
from lewm.eligible_floor_registration_development import bind
from scripts import fit_go2_local_motion_controls_development as previous
from scripts import prepare_go2_short_pulse_training_development as source

OUTPUT=source.BASE/'go2_short_pulse_command_control_v1_attempt_001'
dataset=bind(previous.dataset,ROOTS=source.ROOTS)


def features(rows):
    # The command-only column selection never consumes visual-motion columns.
    return {r['sample_id']:dict(history_available=False,history_features=None,
        data_role=r['data_role'],measured_ns=r['observation_horizon_receipt']['departure_ns'],
        history_frames=r['observation_horizon_receipt']['history_observation_indices']) for r in rows}


def main():
    if OUTPUT.exists():raise ValueError('preserve pulse command-control fit')
    schedule_path=source.OUTPUT/'schedule.json';schedule=json.loads(schedule_path.read_text())
    training=source.load_training_rows();train=dataset(training,features(training))
    weights=np.asarray([schedule['context_draw_counts'][r['sample_id']] for r in training])
    columns=np.arange(13,460)
    model=previous.fit(train,weights,columns)
    with np.load(previous.OUTPUT/'command_only.npz',allow_pickle=False) as archive:
        old_model={k:archive[k].copy() for k in archive.files}
    old=[r for r in json.loads((previous.TRANSFER/'windows.json').read_text()) if r['available']]
    new=[r for r in json.loads((source.PULSE/'windows.json').read_text()) if r['available'] and r['data_role']=='geometry_transfer']
    rows=old+new;data=dataset(rows,features(rows))
    predictions=dict(pulse_augmented=previous.predict(data,model,columns),
        original=previous.predict(data,old_model,columns),command_integrated=data['nominal'])
    masks=dict(original_transfer=np.arange(len(rows))<len(old),short_pulse_transfer=np.arange(len(rows))>=len(old),
        planner_departure=np.asarray([r['source']=='short_pulse' and r['observation_horizon_receipt']['departure_tick']==13 for r in rows]))
    for cluster in ('cluster_02','cluster_03'):
        masks['pulse_'+cluster]=np.asarray([r['source']=='short_pulse' and r['cluster']==cluster for r in rows])
    result=dict(status='COMPLETE',training_contexts=len(training),training_draws=int(weights.sum()),
        ridge_penalty=1.,pulse_training_contexts=180,original_transfer_contexts=len(old),pulse_transfer_contexts=len(new),
        scores={n:{g:previous.scores(p,data,m) for g,m in masks.items()} for n,p in predictions.items()},
        schedule_sha256=hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
        old_model_sha256=hashlib.sha256((previous.OUTPUT/'command_only.npz').read_bytes()).hexdigest(),
        inference_uses_only_past_and_prospective_commands=True,closed_loop_tested=False,
        fit_completed_before_transfer_labels_materialized=True)
    OUTPUT.mkdir();np.savez_compressed(OUTPUT/'command_only.npz',**model)
    np.savez_compressed(OUTPUT/'transfer_predictions.npz',**predictions)
    (OUTPUT/'sample_ids.json').write_text(json.dumps([r['sample_id'] for r in rows])+'\n')
    (OUTPUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
