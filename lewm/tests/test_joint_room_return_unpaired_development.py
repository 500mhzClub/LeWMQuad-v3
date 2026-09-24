"""Unpaired reporting preserves every original raw, replay and native hold gate."""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from scripts import read_go2_joint_room_return_unpaired_v1 as new
from scripts import audit_go2_joint_room_return_v1 as old


def fn(path,name):
    return deepcopy(next(n for n in ast.parse(Path(path).read_text()).body
        if isinstance(n,ast.FunctionDef) and n.name==name))


def test_only_pairing_interpretation_output_root_and_success_label_differ():
    class Normalize(ast.NodeTransformer):
        def visit_Name(self,n):
            if n.id=='INPUT':n.id='OUTPUT'
            return n
        def visit_Assign(self,n):
            if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='describe_setup':
                return ast.Expr(value=ast.Call(func=ast.Name(id='verify_paired_setup',ctx=ast.Load()),args=n.value.args,keywords=[]))
            return self.generic_visit(n)
        def visit_Call(self,n):
            if isinstance(n.func,ast.Name) and n.func.id=='dict':
                n.keywords=[k for k in n.keywords if k.arg!='pairing']
                for k in n.keywords:
                    if k.arg=='paired_setup_prefix_and_first_rgb_verified':k.value=ast.Constant(value=True)
                    if k.arg=='unpaired_native_full_return_success':k.arg='full_room_return_success'
            return self.generic_visit(n)
    a=fn(old.__file__,'audit_condition');b=fn(new.__file__,'read_condition');b.name=a.name
    assert ast.dump(a)==ast.dump(Normalize().visit(b))
    assert ast.dump(fn(old.__file__,'verify_timings'))==ast.dump(fn(new.__file__,'verify_timings'))
    assert new.audit_sensors is old.audit_sensors


@pytest.mark.parametrize('native_diff',[False,True])
def test_rgb_inequality_is_retained_even_when_native_setup_matches(tmp_path,native_diff):
    a=tmp_path/'fresh';b=tmp_path/'old';a.mkdir();b.mkdir()
    x=np.zeros((480,640,3),np.uint8);y=x.copy();y[2,3]=[10,20,30]
    for d,r in ((a,x),(b,y)):
        Image.fromarray(r).save(d/'rgb_0000.png')
        np.savez_compressed(d/'native_depth_0000.npz',optical_depth_m=np.ones((480,640),np.float32))
    names=('base_pose_world','base_twist_world','joint_position','joint_velocity','requested_command','applied_command')
    raw={k:np.zeros((750,3)) for k in names};baseline=deepcopy(raw)
    if native_diff:baseline['joint_position'][3,0]=1
    c=[dict(rgb_sha256=hashlib.sha256(x.tobytes()).hexdigest())]
    oc=[dict(rgb_sha256=hashlib.sha256(y.tobytes()).hexdigest())]
    r=new.describe_setup(raw,c,baseline,oc,directory=a,predecessor=b)
    assert r['native_prefix_exact']==(not native_diff)
    assert not r['first_rgb_exact'] and not r['paired_setup_prefix_and_first_rgb_verified']
    assert r['different_rgb_pixels']==1 and r['different_rgb_channels']==3
    assert r['maximum_rgb_channel_difference']==30 and not r['original_pairing_criterion_relaxed']


def test_new_root_cannot_overwrite_original_audit_or_raw_data(monkeypatch,tmp_path):
    assert new.OUTPUT!=new.INPUT!=new.PREVIOUS
    monkeypatch.setattr(new,'OUTPUT',tmp_path)
    with pytest.raises(ValueError,match='explicit output'):new.metadata('raw_return_audit.json',{})
    with pytest.raises(ValueError,match='explicit output'):new.metadata('rgb_0000.png',{})
    new.metadata('launch.json',dict(original_paired_audit_remains_failed=True))
    with pytest.raises(FileExistsError):new.metadata('launch.json',{})
    with pytest.raises(ValueError,match='exact nonsymlink'):new.main()
    monkeypatch.setattr(new,'validate_root',lambda *a,**k:None)
    with pytest.raises(ValueError,match='exclusive'):new.main()
