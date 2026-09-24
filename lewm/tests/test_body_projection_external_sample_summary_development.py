from copy import deepcopy
import pytest
from scripts import body_projection_external_sample_summary_development as analysis

SOURCE='/synthetic/ordinary/markers.py'


def fixture():
    frames=[dict(name='admission',file='/synthetic/ordinary/inputs.py',line=3,col=None),
        dict(name='observe',file='/synthetic/ordinary/controller.py',line=20,col=None),
        dict(name='observe',file='/synthetic/ordinary/controller.py',line=21,col=None)]
    samples=[[0]]
    for frame,name in analysis.MARKERS.items():
        frames.append(dict(name=name,file=SOURCE,line=10,col=None))
        samples.extend([[0,len(frames)-1,1,2],[0,len(frames)-1]])
    return dict(exporter='py-spy@0.4.2',**{'$schema':'https://www.speedscope.app/file-format-schema.json'},
        shared=dict(frames=frames),profiles=[dict(type='sampled',name='Thread 17 ""',unit='seconds',
            startValue=0.,endValue=len(samples)*.01,samples=samples,weights=[.01]*len(samples))])


def summarize(value):
    return analysis.summarize(value,marker_source_file=SOURCE,owner_native_thread_id=17)


def test_all_thirty_frames_are_covered_and_only_descendant_stacks_are_attributed():
    source=fixture();before=deepcopy(source);report=summarize(source)
    assert source==before
    assert report['total_samples']==61 and report['marked_samples']==60 and report['unmarked_samples']==1
    assert report['wrapper_only_samples']==30 and report['marked_observations']==30
    assert report['all_marked_observations_sampled']
    assert not report['cpu_self_time_established'] and not report['nominal_sample_weights_are_measured_durations']
    for window in report['windows'].values():
        assert len(window['observations'])==10
        assert all(row['samples']==2 and row['controller_samples']==1 for row in window['observations'])
        assert window['inclusive_function_samples']==[dict(file='/synthetic/ordinary/controller.py',function='observe',samples=10)]
        assert window['inclusive_file_samples']==[dict(file='/synthetic/ordinary/controller.py',samples=10)]
        assert window['sampled_leaf_locations']==[dict(file='/synthetic/ordinary/controller.py',line=21,function='observe',samples=10)]


@pytest.mark.parametrize('fault',['exporter','schema','unit','thread','weight','nan','weight_count','timeline',
    'negative_index','boolean_index','large_index','empty_stack','nested_marker','unexpected_marker',
    'missing_frame','wrapper_only_frame','protected','frame_line','marker_source','owner_id'])
def test_incomplete_or_malformed_profile_cannot_claim_complete_controller_coverage(fault):
    profile=fixture();thread=profile['profiles'][0]
    if fault=='exporter':profile['exporter']='other'
    elif fault=='schema':profile['$schema']='other'
    elif fault=='unit':thread['unit']='milliseconds'
    elif fault=='thread':thread['name']='Thread 18 ""'
    elif fault=='weight':thread['weights'][0]=.02
    elif fault=='nan':thread['weights'][0]=float('nan')
    elif fault=='weight_count':thread['weights'].pop()
    elif fault=='timeline':thread['endValue']+=.01
    elif fault=='negative_index':thread['samples'][0]=[-1]
    elif fault=='boolean_index':thread['samples'][0]=[True]
    elif fault=='large_index':thread['samples'][0]=[100_000]
    elif fault=='empty_stack':thread['samples'][0]=[]
    elif fault=='nested_marker':thread['samples'][1].insert(1,4)
    elif fault=='unexpected_marker':profile['shared']['frames'][3]['name']='controller_observation_0000'
    elif fault=='missing_frame':thread['samples'][1]=[0];thread['samples'][2]=[0]
    elif fault=='wrapper_only_frame':thread['samples'][1]=thread['samples'][2].copy()
    elif fault=='protected':profile['shared']['frames'][0]['file']='/synthetic/sealed_/never_opened.py'
    elif fault=='frame_line':profile['shared']['frames'][0]['line']=True
    elif fault=='marker_source':
        with pytest.raises(ValueError):analysis.summarize(profile,marker_source_file='/wrong.py',owner_native_thread_id=17)
        return
    elif fault=='owner_id':
        with pytest.raises(ValueError):analysis.summarize(profile,marker_source_file=SOURCE,owner_native_thread_id=True)
        return
    with pytest.raises(ValueError):summarize(profile)
