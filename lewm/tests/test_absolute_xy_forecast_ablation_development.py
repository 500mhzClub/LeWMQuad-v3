import numpy as np
from lewm.absolute_xy_forecast_ablation_development import predict_xy


def fit(columns):
    n=len(columns);rng=np.random.default_rng(9)
    return dict(columns=columns,mean=np.zeros((8,n)),scale=np.ones((8,n)),
        coefficient=rng.normal(size=(8,n,2)),bias=np.zeros((8,2)))


def test_ablated_xy_cannot_depend_on_neural_forecasts():
    prediction=np.zeros((8,5));prediction[:,3]=1.
    altered=prediction.copy();altered[:,:4]=np.random.default_rng(4).normal(size=(8,4))
    past=np.linspace(-.1,.1,12);commands=np.zeros((8,3))
    ablated=fit(np.r_[np.arange(12),np.arange(16,42)])
    np.testing.assert_array_equal(predict_xy(ablated,prediction,past,commands),
        predict_xy(ablated,altered,past,commands))
    control=fit(np.arange(42))
    assert not np.allclose(predict_xy(control,prediction,past,commands),
        predict_xy(control,altered,past,commands))


def test_each_horizon_uses_only_its_known_command_prefix():
    prediction=np.zeros((8,5));prediction[:,3]=1.
    commands=np.zeros((8,3));changed=commands.copy();changed[4:]=[.2,0.,.45]
    for columns in (np.arange(42),np.r_[np.arange(12),np.arange(16,42)]):
        model=fit(columns)
        before=predict_xy(model,prediction,np.zeros(12),commands)
        after=predict_xy(model,prediction,np.zeros(12),changed)
        np.testing.assert_array_equal(before[:4],after[:4])
        assert not np.allclose(before[4:],after[4:])


def test_runtime_replaces_only_xy_and_preserves_neural_heading_contact():
    from threading import Lock
    from types import SimpleNamespace
    from lewm.absolute_xy_forecast_ablation_development import PoseActionXYForecastRuntime
    runtime=object.__new__(PoseActionXYForecastRuntime)
    runtime.absolute_xy_fit=fit(np.r_[np.arange(12),np.arange(16,42)])
    runtime.correction_pose_lock=Lock()
    runtime.correction_poses={f:dict(position_initial_body_m=[.01*f,0.,0.],
        rotation_initial_body_from_current_body=np.eye(3).tolist(),measured_ns=f*100_000_000)
        for f in range(4)}
    prediction=np.random.default_rng(7).normal(size=(6,8,5)).astype(np.float32)
    before=prediction.copy();packet=SimpleNamespace(frame=3,measured_ns=300_000_000)
    corrected,receipt=runtime._correct_prediction(prediction,packet,None,[[0.,0.,0.]]*3)
    np.testing.assert_array_equal(prediction,before)
    np.testing.assert_array_equal(corrected[:,:,2:],prediction[:,:,2:])
    assert not np.allclose(corrected[:,:,:2],prediction[:,:,:2])
    assert receipt['neural_forecast_inputs_to_xy'] is False
    assert receipt['original_motion_residual_applied'] is False
