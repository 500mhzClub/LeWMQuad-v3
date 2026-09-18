"""Check online packet preprocessing against real retained training inputs."""
import copy
from pathlib import Path

import numpy as np
import torch

from lewm.dense_native_observation_development import dense_native_context
from lewm.route_rgb_dataset_development import load_route_observation
from scripts import train_go2_horizon_dense_predictor_development as fit

RESULT=Path('docs/go2_dense_native_observation_check_2026-09-18.json')


def main():
    assert not RESULT.exists();torch.set_num_threads(1)
    rows=[r for r in fit.parent.reference.selected_rows() if r['data_role']=='train']
    records=[];rejections=[]
    for row in (rows[0],rows[-1]):
        directory,expected_control,_=fit.parent.reference.inputs(row)
        packets=[load_route_observation(directory,i) for i in (3,8,13)]
        now=packets[-1]['image']['measured_ns']
        actual=dense_native_context(packets,observed_ns=now)
        expected=torch.stack([fit.parent.reference.encoders.preprocess_vjepa(str(directory/f'rgb_{i:04d}.png')) for i in (3,8,13)])
        assert torch.equal(actual['pixels'],expected)
        np.testing.assert_array_equal(actual['past_applied_commands'].numpy()[:,[0,2]].reshape(3,5,2),expected_control)
        records.append(dict(trial=row['trial'],prefix_action=row['prefix_action'],pixel_max_abs_error=0.,raw_control_exact=True))
    # Reject an otherwise-valid four-frame/100-ms legacy history at the boundary.
    bad=[load_route_observation(directory,i) for i in (11,12,13)]
    cases={'wrong_spacing':bad,'missing_frame':packets[:2]}
    small=copy.deepcopy(packets);small[0]['image']['rgb']=small[0]['image']['rgb'][::5,::5]
    cases['downsampled_rgb']=small
    late=copy.deepcopy(packets);late[-1]['sensor_state']['control']['applied_command']['available_ns'][-1]=now+1
    cases['future_command_availability']=late
    for name,value in cases.items():
        try:dense_native_context(value,observed_ns=now)
        except ValueError as error:rejections.append(dict(case=name,error=str(error)))
        else:raise AssertionError(name)
    result=dict(status='PASS',training_records=records,rejected_inputs=rejections,
        source_sha256={p:fit.digest(p) for p in (__file__,'lewm/dense_native_observation_development.py')},
        encoder_preprocessing_sha256=fit.digest(fit.parent.reference.encoders.__file__),
        gpu_used=False,new_training=False,new_navigation=False,
        interpretation='Native packet adapter exactly matches retained file preprocessing and raw control order; runtime buffering, inference latency and navigation still need integration.')
    fit.save(RESULT,result);print(result,flush=True)


if __name__=='__main__':main()
