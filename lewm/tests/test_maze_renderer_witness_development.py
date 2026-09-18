from copy import deepcopy
import json
from types import SimpleNamespace as NS
import numpy as np
import pytest
from scripts import maze_renderer_witness_development as witness
from scripts.dual_camera_novel_maze_session_development import DualCameraNovelMazeSession
from scripts.renderer_witness_dual_camera_maze_session_development import RendererWitnessDualCameraMazeSession, ARTIFACT


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    calls=[]; values={}; mode={'fault':None}
    def init(self):
        self.ctx=NS(build=NS(camera=NS(uid='camera0',transform=np.eye(4))),runner=NS(_sim_time_ns=1_500_000_000))
        self.samples=[None]*750;self.output=tmp_path
        self.depth_manifest=[];self.depth_audit=[];self.model_manifest=[];self.auxiliary_audit=[]
    def primary(self, output, name):
        calls.append(('primary',name))
        row=dict(rgb_sha256='a'*64);self.depth_audit.append(dict(native_depth_sha256='b'*64))
        values['primary']=row;return row
    def packets(self):
        calls.append('packets');i=(len(self.samples)-750)//50
        if len(self.model_manifest)==i:
            value=self.capture_fixed_rgb(self.output,f'rgb_{i:04d}')
            assert value is values['primary']
            self.depth_manifest.append({});self.model_manifest.append({})
            self.auxiliary_audit.append(dict(rgb_sha256='c'*64,native_depth_sha256='d'*64))
        result=(object(),object(),object(),object(),dict(primary_rgb_sha256='a'*64),self.ctx.runner._sim_time_ns)
        values['packets']=result;return result
    def identity(camera):
        calls.append('identity')
        if mode['fault']=='clock':session.ctx.runner._sim_time_ns+=1
        if mode['fault']=='pose':camera.transform[0,3]+=.01
        if mode['fault']=='query':raise RuntimeError('synthetic query failure')
        return dict(camera_uid='camera0',framebuffer_matches_camera_depth_target=True,
            vendor='synthetic',version='changed' if mode['fault']=='drift' else 'same',
            source_implementation_equivalence_proven=False,raster_error_bound_proven=False)
    monkeypatch.setattr(DualCameraNovelMazeSession,'__init__',init)
    monkeypatch.setattr(DualCameraNovelMazeSession,'capture_fixed_rgb',primary)
    monkeypatch.setattr(DualCameraNovelMazeSession,'sensor_packets',packets)
    monkeypatch.setattr(DualCameraNovelMazeSession,'persist_observations',lambda self,output:calls.append('persist'))
    monkeypatch.setattr(witness,'renderer_identity_readback',identity)
    monkeypatch.setattr(witness,'sampling_readback',lambda camera:deepcopy(witness.SAMPLING))
    monkeypatch.setattr(witness,'precision_readback',lambda camera:dict(depth_target_depth_bits=24))
    session=RendererWitnessDualCameraMazeSession()
    return session,calls,values,mode


def captures(document):
    return [dict(frame=r['frame'],measured_ns=r['measured_ns'],physical_sample_index=r['physical_sample_index'],
        primary_world_from_optical=(np.asarray(r['camera_transform'])@np.diag([1.,-1.,-1.,1.])).tolist(),
        **r['pixel_hashes']) for r in document['paired']]


def test_existing_parent_captures_and_packets_preserved_with_two_endpoint_queries(fixture):
    session,calls,values,mode=fixture
    result=session.sensor_packets()
    assert result is values['packets']
    assert calls==['packets',('primary','rgb_0000'),'identity','identity']
    before=deepcopy(session.renderer_witnesses)
    result=session.sensor_packets()
    assert result is values['packets'] and calls[-1]=='packets'
    assert session.renderer_witnesses==before  # Reusing captured packets does not invent an acquisition.
    session.samples.extend([None]*50);session.ctx.runner._sim_time_ns+=100_000_000
    session.sensor_packets()
    report=witness.audit_witnesses(session.renderer_witnesses,captures(session.renderer_witnesses))
    assert report['frames']==2 and report['capture_endpoints']==4
    assert not report['raster_error_bound_proven'] and not report['visibility_outcomes_replaced']
    session.persist_observations(session.output)
    assert json.loads((session.output/ARTIFACT).read_text())==session.renderer_witnesses


@pytest.mark.parametrize('fault',['clock','pose','query'])
def test_query_failure_or_mutation_latches_and_preserves_partial_evidence(fixture,fault):
    session,calls,values,mode=fixture;mode['fault']=fault
    with pytest.raises((ValueError,RuntimeError)):session.sensor_packets()
    assert len(session.renderer_witnesses['failures'])==1
    previous=list(calls)
    with pytest.raises(ValueError,match='terminal'):session.sensor_packets()
    assert calls==previous
    session.persist_observations(session.output)
    assert json.loads((session.output/ARTIFACT).read_text())['failures']


def test_context_drift_after_auxiliary_capture_is_rejected(fixture,monkeypatch):
    session,calls,values,mode=fixture
    original=DualCameraNovelMazeSession.sensor_packets
    def drift(self):
        result=original(self);mode['fault']='drift';return result
    monkeypatch.setattr(DualCameraNovelMazeSession,'sensor_packets',drift)
    with pytest.raises(ValueError,match='endpoint drift'):session.sensor_packets()
    assert len(session.renderer_witnesses['primary'])==1 and not session.renderer_witnesses['paired']
    assert len(session.renderer_witnesses['failures'])==1


@pytest.mark.parametrize('fault',['missing','hash','sample','pose','drift','new_render','bound_claim'])
def test_audit_rejects_incomplete_or_inconsistent_acquisition_evidence(fixture,fault):
    session,*_=fixture;session.sensor_packets()
    document=deepcopy(session.renderer_witnesses);raw=captures(document)
    if fault=='missing':document['paired'].clear()
    if fault=='hash':raw[0]['auxiliary_rgb_sha256']='e'*64
    if fault=='sample':raw[0]['physical_sample_index']+=1
    if fault=='pose':raw[0]['primary_world_from_optical'][0][3]+=.02
    if fault=='drift':document['paired'][0]['context']['identity']['version']='other'
    if fault=='new_render':document['paired'][0]['render_calls_added']=1
    if fault=='bound_claim':document['paired'][0]['raster_error_bound_proven']=True
    with pytest.raises((ValueError,AssertionError)):witness.audit_witnesses(document,raw)


def test_provenance_is_persisted_even_if_parent_persistence_fails(fixture,monkeypatch):
    session,*_=fixture;session.sensor_packets()
    def failure(self,output):raise OSError('synthetic parent persistence failure')
    monkeypatch.setattr(DualCameraNovelMazeSession,'persist_observations',failure)
    with pytest.raises(OSError):session.persist_observations(session.output)
    assert json.loads((session.output/ARTIFACT).read_text())==session.renderer_witnesses
