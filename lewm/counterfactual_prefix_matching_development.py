"""Exact physical matching with explicit small-render-variation diagnostics."""
import json
from pathlib import Path
import numpy as np
from PIL import Image


def rgb_difference(reference,candidate):
    if reference.shape!=(480,640,3) or candidate.shape!=reference.shape or reference.dtype!=np.uint8 or candidate.dtype!=np.uint8:
        raise ValueError('native RGB identity mismatch')
    delta=reference.astype(np.float64)-candidate.astype(np.float64)
    return {'changed_pixels':int(np.count_nonzero(np.any(delta!=0,axis=2))),
        'changed_pixel_fraction':float(np.mean(np.any(delta!=0,axis=2))),
        'rms_8bit':float(np.sqrt(np.mean(delta**2))),'maximum_channel_difference':int(np.max(np.abs(delta)))}


def compare_prefix(reference_dir,reference,candidate_dir,candidate):
    for row in (reference,candidate):
        if not 0<=row['branch_start_observation_index']<341: raise ValueError('prefix frame index')
    a,b=reference['prefix_binding'],candidate['prefix_binding']
    for key in ('physics_arrays','history_arrays','physics_samples','timestamp_ns'):
        if a[key]!=b[key]: raise ValueError(f'physical/history prefix mismatch: {key}')
    if reference['branch_start_observation_index']!=candidate['branch_start_observation_index']:
        raise ValueError('prefix frame population mismatch')
    reference_dir,candidate_dir=Path(reference_dir),Path(candidate_dir)
    cameras=[json.loads((directory/'camera_audit.json').read_text()) for directory in (reference_dir,candidate_dir)]
    metrics=[]
    for i in range(reference['branch_start_observation_index']+1):
        for key in ('timestamp_s','physical_sample_index','world_from_optical','rigid_mount_no_obstacle_adjustment'):
            if cameras[0][i][key]!=cameras[1][i][key]: raise ValueError(f'camera prefix mismatch: {key}')
        with Image.open(reference_dir/f'rgb_{i:04d}.png') as image: first=np.array(image)
        with Image.open(candidate_dir/f'rgb_{i:04d}.png') as image: second=np.array(image)
        value=rgb_difference(first,second)
        if value['rms_8bit']>1. or value['changed_pixel_fraction']>.001:
            raise ValueError('prefix RGB difference exceeds reviewed tolerance')
        metrics.append(value)
    return {'status':'MATCH','physical_and_history_exact':True,'camera_geometry_exact':True,
        'prefix_frames':len(metrics),'max_rgb_rms_8bit':max(v['rms_8bit'] for v in metrics),
        'max_changed_pixel_fraction':max(v['changed_pixel_fraction'] for v in metrics),
        'canonical_model_context_scene_id':reference['scene_id'],
        'canonical_model_context_observation_index':reference['branch_start_observation_index']}
