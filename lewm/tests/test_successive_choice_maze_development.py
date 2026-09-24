import copy

import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.counterfactual_maze_development import corpus as training_corpus
from lewm.online_choice_maze_pilot_development import corpus as pilot_corpus
from lewm.successive_choice_maze_development import corpus,layout_spec,trials,execute_control


def test_fresh_fixed_population():
    layouts=corpus(); rows=trials()
    assert len(layouts)==8 and len(rows)==144 and len({r['scene_id'] for r in rows})==144
    assert {r['procedural_seed'] for r in layouts}==set(range(2026091800,2026091808))
    assert not {r['topology_sha256_dihedral'] for r in layouts}&{r['topology_sha256_dihedral'] for r in training_corpus()+pilot_corpus()}
    assert rows==trials()
    rows[0]['geometry']['wall_boxes'].clear()
    assert rows[1]['geometry']['wall_boxes']


@pytest.mark.parametrize('index',[-1,8,True,1.2])
def test_panel_bounds(index):
    with pytest.raises(ValueError): layout_spec(index)


class Loop:
    def __init__(self,fail=None):
        self.time=0; self.observed=0; self.steps=[]; self.choices=[]; self.records=[]; self.fail=fail; self.faults=[]
    def select(self,*,now_ns):
        assert now_ns==self.time==self.observed and now_ns%500_000_000==0
        if self.fail=='select' and self.time==500_000_000: raise SensorContractError('select failure')
        return {'decision_index':self.time//500_000_000,'requested_command_tape':[[.2,0,0]]*5}
    def observe(self):
        if self.fail=='observe' and self.time==200_000_000: raise SensorContractError('input failure')
        self.observed=self.time
    def step(self,command,release):
        assert self.records[-1][0]==command
        self.steps.append((copy.deepcopy(command),release)); self.time+=100_000_000
        if self.fail=='native' and self.time==200_000_000: raise RuntimeError('native halt')
    def run(self):
        return execute_control(self,observe=self.observe,step=self.step,clock=lambda:self.time,
            record_selection=self.choices.append,record_command=lambda *args:self.records.append(args),record_fault=self.faults.append)


def test_eight_fresh_choices_and_release():
    loop=Loop(); result=loop.run()
    assert result=={'sensor_fault':None,'completed_control_ticks':40,'selections':8,'release_ticks':5}
    assert len(loop.choices)==8 and len(loop.steps)==45 and loop.observed==4_000_000_000
    assert loop.steps[-5:]==[([0.,0.,0.],True)]*5


@pytest.mark.parametrize('fault,count',[('select',5),('observe',2)])
def test_fault_cancels_motion_and_dispatches_zero(fault,count):
    loop=Loop(fault); result=loop.run()
    assert result['sensor_fault'] and result['completed_control_ticks']==count
    assert loop.faults==[result['sensor_fault']]
    assert len(loop.steps)==count+5 and loop.steps[-5:]==[([0.,0.,0.],True)]*5


def test_native_halt_never_steps_again():
    loop=Loop('native')
    with pytest.raises(RuntimeError,match='native halt'): loop.run()
    assert len(loop.steps)==2 and not any(release for _,release in loop.steps)
