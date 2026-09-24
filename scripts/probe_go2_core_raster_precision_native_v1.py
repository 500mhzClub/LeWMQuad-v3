"""Zero-scene native FBO query comparison; no rendering or physics rerun."""
import json
import shutil
from types import SimpleNamespace
from lewm_genesis.core_raster_precision_development import precision_readback
from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
from scripts.ordered_dynamic_session_development import precision_readback as old_readback
from scripts.run_go2_ordered_union_dynamic_sensor_pilot_v1 import OUTPUT as FAILED
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch
from scripts.startup_source_inventory_development import discover_sources
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json

OUTPUT=BASE/'go2_core_raster_precision_native_probe_v1_attempt_001'
PROTOCOL='docs/go2_core_raster_precision_native_probe_v1_2026-09-06.md'
IDS={'launch.json':'ac2542dfab74b5c4df0f7ad4797999e400c0a6d58b2e11d52db87e777e16d626',
     'failure.json':'ace19931d193ad78054090b0f580e029a4005bb9481fb64ed6bca3961c9dce96',
     'dynamic_audit.json':'404a10fdcb7dfdb16553c6f5a4f4e4efa812af3edf706334a014e5457041fd94'}


def main():
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive native query bench')
    verify_artifacts(FAILED,IDS);old=read_json(FAILED,'launch.json');verify_ordered_launch(old)
    launch={k:old[k] for k in ('input_sha256','native_sha256','native_scene_sha256','native_geometry_sha256',
        'opencv_binary_sha256','opencv_version','rules')}
    launch.update(source_sha256=discover_sources((PROTOCOL,'scripts/probe_go2_core_raster_precision_native_v1.py',
        'lewm/tests/test_core_raster_precision_development.py'),old['source_sha256']),predecessor_sha256=IDS,
        maximum_artifact_bytes=32*1024**2,minimum_free_bytes=40*1024**3,scenes=0,physics_steps=0,rendered_frames=0)
    verify_ordered_launch(launch)
    if len((json.dumps(launch,indent=2)+'\n').encode())>16*1024**2:raise ValueError('metadata budget')
    if shutil.disk_usage(BASE.parent).free<launch['minimum_free_bytes']+launch['maximum_artifact_bytes']:raise ValueError('reserve')
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    context=target=None
    try:
        initialize_genesis(backend='cpu',seed=2026090601,logging_level='warning')
        from genesis.ext.pyrender.offscreen import OffscreenRenderer
        from genesis.ext.pyrender.renderer import Renderer
        from OpenGL.GL import (glBindFramebuffer,glGetIntegerv,glGetString,glGetError,GL_DRAW_FRAMEBUFFER,
            GL_DRAW_FRAMEBUFFER_BINDING,GL_RENDERER,GL_VERSION,GL_CONTEXT_PROFILE_MASK,GL_CONTEXT_CORE_PROFILE_BIT,GL_NO_ERROR)
        context=OffscreenRenderer(pyopengl_platform='egl',seg_node_map={});context.make_current()
        try:
            target=Renderer(640,480,None);target._configure_main_framebuffer()
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER,target._main_fb)
            native=dict(renderer=glGetString(GL_RENDERER).decode(),version=glGetString(GL_VERSION).decode(),
                profile_mask=int(glGetIntegerv(GL_CONTEXT_PROFILE_MASK)))
        finally:context.make_uncurrent()
        camera=SimpleNamespace(uid=0,_rasterizer=SimpleNamespace(_renderer=context,_camera_targets={0:target}))
        old_error=None;old_value=None
        try:old_value=old_readback(camera)
        except Exception as error:old_error=dict(type=type(error).__name__,message=str(error),code=int(getattr(error,'err',-1)))
        new_value=precision_readback(camera)
        context.make_current()
        try:
            restored=int(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING))==int(target._main_fb)
            clean=glGetError()==GL_NO_ERROR
        finally:context.make_uncurrent()
        verify_ordered_launch(launch);verify_artifacts(FAILED,IDS)
        result=dict(status='CORE_RASTER_PRECISION_NATIVE_PROBE_COMPLETE',native=native,old_query_error=old_error,
            old_query_value=old_value,corrected_query=new_value,depth_framebuffer_restored=restored,error_state_clean=bool(clean),
            core_profile=bool(native['profile_mask']&int(GL_CONTEXT_CORE_PROFILE_BIT)),
            expected_mechanism_observed=bool(old_error and old_error['code']==1280 and restored and clean),
            scenes=0,physics_steps=0,rendered_frames=0,collection_retry_authorized=False,navigation_qualified=False,goal_achieved=False)
        write_json(OUTPUT/'result.json',result)
        print('CORE_PRECISION_PROBE',result,flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='CORE_PRECISION_NATIVE_PROBE_FAILURE',reason=str(error)))
        raise
    finally:
        if context is not None:
            if target is not None:
                context.make_current()
                try:target.delete()
                finally:context.make_uncurrent()
            context.delete()
        shutdown_genesis()


if __name__=='__main__':main()
