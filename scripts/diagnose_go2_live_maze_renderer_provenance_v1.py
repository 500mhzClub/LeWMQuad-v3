"""Read live DRM accounting and a separately scoped fresh EGL context identity."""
import hashlib
import json
from pathlib import Path
import re
import time
from types import SimpleNamespace
import psutil
from lewm_genesis.camera_renderer_identity_development import renderer_identity_readback
from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT = BASE/'go2_live_maze_renderer_provenance_v1_attempt_001'
INPUT = BASE/'go2_later_floor_resolution_maze_pilot_v1_attempt_001'
PRIOR = BASE/'go2_core_raster_precision_native_probe_v1_attempt_001'
INPUT_SHA = 'c3d035abcc69b3b42ecb160021203e7d6d685a176e860e5c3044afa2035cefa4'
PRIOR_SHA = 'fc2d51f3011294573247cfb1782f9c0631dca8a1daa1a8ae390af2e70db60819'
PROTOCOL = 'docs/go2_live_maze_renderer_provenance_v1_2026-09-09.md'
WORKER = 2398270
PARENT = 2398198
EGL_SOURCE = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/lib/python3.12/site-packages/genesis/ext/pyrender/platforms/egl.py')
LIBRARIES = tuple(Path('/usr/lib/x86_64-linux-gnu')/n for n in (
    'libEGL_mesa.so.0.0.0', 'libgallium-25.2.8-0ubuntu0.24.04.2.so', 'libLLVM.so.20.1'))


def process_identity():
    p = psutil.Process(WORKER); parent = psutil.Process(PARENT)
    if (not p.is_running() or p.ppid() != PARENT
            or 'scripts/run_go2_later_floor_resolution_maze_pilot_v1.py' not in parent.cmdline()
            or '--multiprocessing-fork' not in p.cmdline()):
        raise ValueError('the same live ninth-native worker and launch parent required')
    return dict(worker_pid=WORKER, worker_create_time=p.create_time(),
        parent_pid=PARENT, parent_create_time=parent.create_time())


def snapshot():
    identity = process_identity(); clients = {}
    for path in sorted((Path('/proc')/str(WORKER)/'fdinfo').iterdir()):
        try: lines = path.read_text().splitlines()
        except FileNotFoundError: continue
        fields = dict(line.split(':',1) for line in lines if ':' in line)
        fields = {k:v.strip() for k,v in fields.items()}
        if 'drm-driver' not in fields: continue
        pci = fields['drm-pdev']; cid = fields['drm-client-id']
        if not re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',pci) or not cid.isdigit():
            raise ValueError('bounded kernel DRM client identity required')
        key = pci+'/'+cid
        client = clients.setdefault(key, dict(driver=fields['drm-driver'], pci=pci, client_id=cid, fds=[], engine_ns={}))
        if client['driver'] != fields['drm-driver']: raise ValueError('consistent duplicate DRM descriptors required')
        client['fds'].append(int(path.name))
        for name,value in fields.items():
            if name.startswith('drm-engine-'):
                if not re.fullmatch(r'[0-9]+ ns',value): raise ValueError('integer nanosecond engine accounting required')
                client['engine_ns'][name] = max(client['engine_ns'].get(name,0),int(value.split()[0]))
        sysfs = Path('/sys/bus/pci/devices')/pci
        client['pci_identity'] = {name:(sysfs/name).read_text().strip() for name in
            ('vendor','device','subsystem_vendor','subsystem_device')}
    paths = {m.path for m in psutil.Process(WORKER).memory_maps(grouped=True)}
    mapped = {str(p):digest(p) for p in LIBRARIES if str(p) in paths}
    if set(mapped) != set(map(str,LIBRARIES)): raise ValueError('all declared live graphics libraries must be mapped')
    timing = INPUT/'full_jepa_novel_maze_00/decision_stream_timing.jsonl'
    data = timing.read_bytes(); rows = data.split(b'\n')[:-1]
    last = json.loads(rows[-1]) if rows else None
    if process_identity() != identity: raise ValueError('worker identity changed during snapshot')
    return dict(identity=identity, monotonic_s=time.monotonic(), drm_clients=clients,
        mapped_library_sha256=mapped, completed_timing_rows=len(rows), latest_timing=last,
        latest_complete_timing_line_sha256=hashlib.sha256(rows[-1]).hexdigest() if rows else None)


def fresh_context():
    context = target = None
    initialize_genesis(backend='cpu',seed=2026090901,logging_level='warning')
    try:
        from genesis.ext.pyrender.offscreen import OffscreenRenderer
        from genesis.ext.pyrender.renderer import Renderer
        from OpenGL.GL import glBindFramebuffer, GL_DRAW_FRAMEBUFFER
        context = OffscreenRenderer(pyopengl_platform='egl',seg_node_map={})
        context.make_current()
        try:
            target = Renderer(640,480,None); target._configure_main_framebuffer()
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER,target._main_fb)
        finally: context.make_uncurrent()
        camera = SimpleNamespace(uid=0,_rasterizer=SimpleNamespace(_renderer=context,_camera_targets={0:target}))
        identity = renderer_identity_readback(camera)
        paths = {m.path for m in psutil.Process().memory_maps(grouped=True)}
        mapped = {str(p):digest(p) for p in LIBRARIES if str(p) in paths}
        if set(mapped) != set(map(str,LIBRARIES)): raise ValueError('declared query-context graphics libraries must be mapped')
        device = identity['egl_device_name']; pci = None
        if device is not None and re.fullmatch(r'/dev/dri/card[0-9]+',device):
            pci = (Path('/sys/class/drm')/Path(device).name/'device').resolve().name
        return dict(identity=identity, mapped_library_sha256=mapped, egl_device_pci=pci,
            live_worker_context_queried=False,
            live_worker_renderer_equivalence_proven=False, historical_renderer_identity_inferred=False,
            scenes=0, physics_steps=0, draw_calls=0, rendered_frames=0)
    finally:
        if context is not None:
            if target is not None:
                context.make_current()
                try: target.delete()
                finally: context.make_uncurrent()
            context.delete()
        shutdown_genesis()


def verify_all(launch):
    verify(launch)
    verify_artifacts(INPUT,{'launch.json':INPUT_SHA})
    verify_artifacts(PRIOR,{'result.json':PRIOR_SHA})
    for name,sha in launch['additional_runtime_sha256'].items():
        if digest(Path(name)) != sha: raise ValueError('additional runtime identity changed: '+name)


def main():
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive renderer provenance diagnosis required')
    verify_artifacts(INPUT,{'launch.json':INPUT_SHA}); old=read_json(INPUT,'launch.json'); verify(old)
    verify_artifacts(PRIOR,{'result.json':PRIOR_SHA}); prior=read_json(PRIOR,'result.json')
    sources=discover_sources((PROTOCOL,'scripts/diagnose_go2_live_maze_renderer_provenance_v1.py',
        'lewm/tests/test_camera_renderer_identity_development.py',
        'docs/go2_llvmpipe_subpixel_source_review_2026-09-08.md'),old['source_sha256'])
    resources=hardware()
    if resources['memory_available_bytes'] < 4*1024**3 or resources['artifact_free_bytes'] < 40*1024**3+32*1024**2:
        raise ValueError('bounded zero-scene query resources unavailable')
    launch=old|dict(protocol=PROTOCOL,output_root=str(OUTPUT),source_sha256=sources,hardware=resources,
        observed_process_identity=process_identity(),additional_runtime_sha256={str(p):digest(p) for p in (*LIBRARIES,EGL_SOURCE)},
        input_launch_sha256=INPUT_SHA,prior_empty_context_result_sha256=PRIOR_SHA,
        native_execution=False,native_context_query=True,scenes=0,physics_steps=0,rendered_frames=0,
        native_scene_workers=0,observed_existing_scene_workers=1,model_loaded=False,model_training=False,
        implementation_class='renderer_identity_readback_on_separate_empty_context',
        cpu_processes=1,numerical_threads=1,output_allowance_bytes=32*1024**2,
        concurrency_reason='bounded zero-scene context query and read-only kernel metadata beside one native scene')
    verify_all(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('LIVE_MAZE_RENDERER_PROVENANCE_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    try:
        before=snapshot();time.sleep(3);after=snapshot()
        if before['identity'] != after['identity'] or before['identity'] != launch['observed_process_identity']:
            raise ValueError('same live process identity required across observation interval')
        deltas={}
        for key,client in before['drm_clients'].items():
            latest=after['drm_clients'][key]; engine={}
            for name,value in client['engine_ns'].items():
                delta=latest['engine_ns'][name]-value
                if delta < 0: raise ValueError('monotone same-client engine accounting required')
                engine[name]=delta
            deltas[key]=engine
        queried=fresh_context();verify_all(launch)
        if process_identity()!=launch['observed_process_identity']:raise ValueError('live worker identity changed')
        write_json(OUTPUT/'result.json',dict(status='LIVE_MAZE_RENDERER_PROVENANCE_COMPLETE',
            launch_sha256=digest(OUTPUT/'launch.json'),source_sha256=sources,
            before=before,after=after,engine_delta_ns=deltas,fresh_context=queried,
            prior_empty_context_native=prior['native'],
            prior_empty_context_identifies_later_maze_renderer=False,
            live_worker_context_queried=False,live_worker_modified=False,
            original_visibility_outcomes_unchanged=True,navigation_qualified=False,goal_achieved=False))
        print('LIVE_MAZE_RENDERER_PROVENANCE_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='LIVE_MAZE_RENDERER_PROVENANCE_FAILURE',reason=repr(error)))
        raise


if __name__=='__main__':main()
