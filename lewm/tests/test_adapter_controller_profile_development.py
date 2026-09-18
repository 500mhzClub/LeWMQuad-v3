import pytest
from scripts.profile_go2_adapter_controller_windows_v1 import profile_summary, resources_for


def test_cumulative_time_is_not_double_counted_as_exclusive_runtime():
    r=profile_summary({('parent.py',1,'parent'):(1,1,.2,1.,{}),('child.py',2,'child'):(2,2,.8,.8,{})})
    assert r['total_exclusive_profiled_s']==1.
    assert r['modules_by_exclusive_time'][0]=={'filename':'child.py','self_time_s':.8}
    assert r['functions'][0]['function']=='parent'
    assert r['cumulative_times_overlap'] and r['cumulative_times_must_not_be_summed']


@pytest.mark.parametrize('filename',['sealed_test.json','data/sealed/source.py','data/sealed_future/source.py'])
def test_protected_path_names_rejected_before_serialization(filename):
    with pytest.raises(ValueError):profile_summary({(filename,1,'f'):(1,1,.1,.1,{})})


@pytest.mark.parametrize('entry',[(2,1,.1,.1,{}),(1,1,-.1,.1,{}),(1,1,.1,float('nan'),{})])
def test_invalid_profiler_measurements_rejected(entry):
    with pytest.raises(ValueError):profile_summary({('source.py',1,'f'):entry})


@pytest.mark.parametrize('memory,storage,cpus',[(47,100,16),(64,40,16),(64,100,2)])
def test_concurrent_profile_requires_declared_resource_headroom(memory,storage,cpus):
    with pytest.raises(ValueError):resources_for(dict(memory_available_bytes=memory*1024**3,
        artifact_free_bytes=storage*1024**3,physical_cpus=cpus))
