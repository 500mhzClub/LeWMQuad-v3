import numpy as np
from types import SimpleNamespace
from threading import Lock
from lewm.short_pulse_navigation_runtime_development import PulsePredictiveRuntime,command_predictions,past_commands
from lewm.terminal_translation_pulse_development import command_sequences


def packets():
    return [dict(sensor_state=dict(decision_ns=1_500_000_000+i*100_000_000,
        control=dict(applied_command=dict(values=np.tile([.16,0.,.45],(15,1)),valid=np.ones((15,3),bool),
            measured_ns=1_500_000_000+i*100_000_000-np.arange(14,-1,-1)*100_000_000,
            available_ns=1_500_000_000+i*100_000_000-np.arange(14,-1,-1)*100_000_000)))) for i in range(4)]


def test_command_forecast_horizon_is_causal_and_missing_clock_rejected():
    model=dict(mean=np.zeros((8,447)),scale=np.ones((8,447)),bias=np.zeros((8,3)),coefficient=np.zeros((8,447,3)))
    # Deliberately make all future-command columns influential; masking must
    # still keep the eighth command out of every earlier prediction.
    model['coefficient'][:,:24,:]=.02
    history=past_commands(packets(),1_800_000_000)
    commands=command_sequences([[0.,0.,0.]]*3,pulse=True)
    a=command_predictions(model,history,commands)
    commands[:,7]=[.2,0.,.45]
    b=command_predictions(model,history,commands)
    np.testing.assert_array_equal(a[:,:7],b[:,:7])
    assert np.any(a[:,7]!=b[:,7])
    import pytest
    future=packets();future[0]['sensor_state']['control']['applied_command']['available_ns'][-1]+=1
    with pytest.raises(ValueError,match='future command'):past_commands(future,1_800_000_000)


def test_predictor_selection_preserves_raw_neural_xy_and_disables_contact():
    runtime=object.__new__(PulsePredictiveRuntime)
    runtime.correction_pose_lock=Lock()
    runtime.correction_poses={i:dict(measured_ns=1_500_000_000+i*100_000_000) for i in range(4)}
    xy=np.full((6,8,2),.031)
    runtime.pose_command_xy=SimpleNamespace(predict=lambda *a,**kw:xy)
    runtime.planning_translation_pulse=True;runtime.command_fit_sha256='test'
    runtime.command_model=dict(mean=np.zeros((8,447)),scale=np.ones((8,447)),bias=np.zeros((8,3)),coefficient=np.zeros((8,447,3)))
    packet=SimpleNamespace(frame=3,measured_ns=1_800_000_000,history=packets())
    prediction=np.zeros((6,8,5));prediction[:,:,:2]=.013;prediction[:,:,3]=1.
    original=prediction.copy()
    for source in ('neural','pose_command','command_history'):
        runtime.pulse_prediction_source=source
        result,receipt=runtime._correct_prediction(prediction,packet,None,[[0.,0.,0.]]*3)
        assert np.all(result[:,:,4]==-1000.)
        assert not receipt['external_neural_correction_applied']
        assert receipt['neural_outcomes_used_for_scoring']==(source=='neural')
        if source=='neural':np.testing.assert_array_equal(result[:,:,:4],prediction[:,:,:4])
        elif source=='pose_command':np.testing.assert_array_equal(result[:,:,:2],xy)
        else:np.testing.assert_allclose(result[1,6,:2],[.02,0.],atol=1e-8)
    np.testing.assert_array_equal(prediction,original)
