import math

import numpy as np
import pytest

from scripts.run_go2_causal_rgb_body_capture_development_v1 import COMMANDS,probe_spec,reduce_response


def test_nine_fixed_probes_are_symmetric_and_have_fresh_ids():
    specs=[probe_spec(i) for i in range(9)]
    assert len({s['scene_id'] for s in specs})==9
    assert [s['procedural_seed'] for s in specs]==list(range(2026090900,2026090909))
    commands={tuple(s['stimulus_command']) for s in specs}
    assert all(tuple(-np.array(cmd)) in commands for cmd in commands)
    assert all(s['geometry']['wall_boxes']==specs[0]['geometry']['wall_boxes'] for s in specs)
    assert len(specs[0]['geometry']['wall_boxes'])==4


def arrays():
    return {'phase':np.array([0]*750+[1]*1500+[2]*750),
        'base_pose_world':np.tile([0,0,.3,0,0,math.sin(math.pi/4),math.cos(math.pi/4)],(3000,1)),
        'base_twist_world':np.zeros((3000,6)), 'physics_contact':np.zeros(3000,dtype=bool)}


def test_response_uses_body_frame_and_last_excitation_second():
    data=arrays()
    data['base_twist_world'][750:1750,1]=.8
    data['base_twist_world'][1750:2250,1]=.2
    result=reduce_response(data,[.2,0,0],None)
    assert result['mean_body_velocity_mps']==pytest.approx([.2,0,0],abs=1e-12)
    assert result['mean_forward_error_mps']==pytest.approx(0,abs=1e-12)
    assert result['completed_fixed_tape'] and result['release_motion_window_pass']


def test_release_window_cannot_hide_a_fast_sample():
    data=arrays()
    data['base_twist_world'][-99,0]=.11
    assert not reduce_response(data,[0,0,0],None)['release_motion_window_pass']


def test_physical_stop_cannot_report_completed_command_tape():
    data=arrays()
    data['physics_contact'][-1]=True
    result=reduce_response(data,[0,0,0],'DISALLOWED_CONTACT')
    assert result['contact'] and not result['completed_fixed_tape']
