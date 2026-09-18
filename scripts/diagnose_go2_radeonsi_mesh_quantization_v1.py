"""Compare every source-listed grid hypothesis on an unchanged completed mesh."""
import hashlib
import numpy as np
import trimesh
from lewm.mesh_subpixel_coverage_diagnosis_development import diagnose_mesh
from scripts.diagnose_go2_mesh_subpixel_coverage_v1 import INPUT, INPUT_SHA, EDGE, EDGE_SHA, CASE
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_radeonsi_mesh_quantization_diagnosis_v1_attempt_001'
MESH = BASE/'go2_mesh_subpixel_coverage_v1_attempt_001'
PROVENANCE = BASE/'go2_live_maze_renderer_provenance_v1_attempt_001'
PROTOCOL = 'docs/go2_radeonsi_mesh_quantization_diagnosis_v1_2026-09-09.md'
RESULTS = {INPUT:INPUT_SHA, EDGE:EDGE_SHA,
    MESH:'8f0eaccafbdde8406665fd7758ee4ccbd0ab22ea1fe78112f076bf778e49788a',
    PROVENANCE:'76a592ec2bfa31927acb07cfdc6bb056de78df1693ab4066477f20a15159150a'}


def verify_all(launch):
    verify(launch)
    for root, bindings in launch['completed_input_bindings'].items(): verify_artifacts(root,bindings)


def main():
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive three-mode mesh diagnosis required')
    inputs={};bindings={};inherited={}
    for root,sha in RESULTS.items():
        verify_artifacts(root,{'result.json':sha}); r=read_json(root,'result.json');inputs[root]=r
        ids={'result.json':sha,**r.get('artifact_sha256',{})}
        if 'launch_sha256' in r:ids['launch.json']=r['launch_sha256']
        verify_artifacts(root,ids);bindings[str(root)]=ids
        for name,value in r['source_sha256'].items():
            if name in inherited and inherited[name]!=value:raise ValueError('incompatible frozen input sources')
            inherited[name]=value
    assert inputs[INPUT]['status']=='VIEW_REENTRY_MAZE_PILOT_COMPLETE'
    assert inputs[MESH]['status']=='MESH_SUBPIXEL_COVERAGE_DIAGNOSIS_COMPLETE'
    assert inputs[PROVENANCE]['status']=='LIVE_MAZE_RENDERER_PROVENANCE_COMPLETE'
    sources=discover_sources((PROTOCOL,'scripts/diagnose_go2_radeonsi_mesh_quantization_v1.py',
        'lewm/tests/test_mesh_subpixel_coverage_diagnosis_development.py',
        'docs/go2_live_maze_renderer_provenance_result_2026-09-09.md'),inherited)
    old=read_json(INPUT,'launch.json');resources=hardware()
    if resources['memory_available_bytes']<8*1024**3 or resources['artifact_free_bytes']<40*1024**3+128*1024**2:
        raise ValueError('bounded geometry diagnosis resources unavailable')
    launch=old|dict(protocol=PROTOCOL,output_root=str(OUTPUT),source_sha256=sources,hardware=resources,
        completed_input_bindings=bindings,subpixel_bit_hypotheses=[8,10,12],mode_selection_performed=False,
        native_execution=False,native_context_query=False,model_loaded=False,model_training=False,
        native_scene_workers=0,cpu_processes=1,numerical_threads=1,
        concurrency_reason='small completed-mesh readout beside the existing separately owned native scene')
    verify_all(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('RADEONSI_MESH_QUANTIZATION_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    try:
        directory=INPUT/CASE[0]
        transform=read_json(directory,'camera_audit.json')[909]['world_from_optical']
        mesh=trimesh.load(directory/'visual_meshes/wall_union_visual.ply',process=False)
        vertices=np.asarray(mesh.vertices,dtype=np.float32);faces=np.asarray(mesh.faces)
        assert hashlib.sha256(vertices.tobytes()).hexdigest()==inputs[MESH]['report']['native_input_positions_sha256']
        with np.load(directory/'native_depth_0909.npz',allow_pickle=False) as z: native=float(z['optical_depth_m'][260,428])
        assert native==inputs[MESH]['report']['native_optical_depth_m']
        reports={}
        for bits in launch['subpixel_bit_hypotheses']:
            report=diagnose_mesh(vertices,faces,transform,[260,428],subpixel_bits=bits,near_m=.005)
            if bits==8:assert report==inputs[MESH]['report']['mesh_report']
            nearest=report['hypothetical_snapped_projection']['nearest_candidates']
            depth=nearest[0]['unsnapped_plane_depth_m'] if nearest else None
            reports[str(bits)]=dict(mesh_report=report,nearest_original_plane_minus_native_m=None if depth is None else depth-native)
        verify_all(launch)
        write_json(OUTPUT/'result.json',dict(status='RADEONSI_MESH_QUANTIZATION_DIAGNOSIS_COMPLETE',
            launch_sha256=digest(OUTPUT/'launch.json'),source_sha256=sources,report=dict(
                hypotheses=reports,eight_bit_predecessor_exact=True,native_optical_depth_m=native,
                original_strict_score=inputs[MESH]['report']['original_strict_score'],
                original_strict_failure_unchanged=True,mode_selection_performed=False,
                active_hardware_quantization_mode_proven=False,complete_visibility_evaluation=False,
                native_pixels_changed=False),native_execution=False,navigation_qualified=False,goal_achieved=False))
        print('RADEONSI_MESH_QUANTIZATION_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='RADEONSI_MESH_QUANTIZATION_DIAGNOSIS_FAILURE',reason=repr(error)))
        raise


if __name__=='__main__':main()
