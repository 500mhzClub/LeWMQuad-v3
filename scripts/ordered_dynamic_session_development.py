"""Mounted native capture with explicit order and raster precision witnesses."""
import numpy as np
from lewm.independent_layout_collection_development import CollectionInventory
from lewm_genesis.ordered_union_raster_development import verify_order
from scripts.ordered_dynamic_physical_init_development import OrderedDynamicPhysicalInit
from scripts.independent_pulse_context_session_development import PulseContextSession
from scripts.near_field_rgbd_capture_development import NearFieldCapture
from scripts.run_go2_successive_choice_maze_development_v1 import write_json


def precision_readback(camera):
    from OpenGL.GL import (glGetIntegerv,glBindFramebuffer,glGetMultisamplefv,
        GL_DRAW_FRAMEBUFFER,GL_DRAW_FRAMEBUFFER_BINDING,GL_SUBPIXEL_BITS,
        GL_DEPTH_BITS,GL_SAMPLES,GL_SAMPLE_POSITION)
    raster=camera._rasterizer;context=raster._renderer;target=raster._camera_targets[camera.uid]
    previous=None;context.make_current()
    try:
        previous=int(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING))
        if previous!=int(target._main_fb):raise ValueError('depth single-sample framebuffer must be current')
        bits=int(glGetIntegerv(GL_SUBPIXEL_BITS));depth_bits=int(glGetIntegerv(GL_DEPTH_BITS))
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER,int(target._main_fb_ms))
        samples=int(glGetIntegerv(GL_SAMPLES))
        if not 1<=bits<=32 or not 1<=depth_bits<=64 or not 1<=samples<=32:
            raise ValueError('bounded native raster precision and RGB multisampling required')
        positions=[np.asarray(glGetMultisamplefv(GL_SAMPLE_POSITION,i),float).reshape(2).tolist() for i in range(samples)]
        if not np.isfinite(positions).all() or np.min(positions)<0 or np.max(positions)>1:
            raise ValueError('finite pixel-local multisample positions required')
        return dict(subpixel_bits=bits,depth_target_depth_bits=depth_bits,rgb_target_samples=samples,
            rgb_target_sample_positions=positions,scope='native target readback after RGB/depth capture; not a precision-error bound')
    finally:
        try:
            if previous is not None:glBindFramebuffer(GL_DRAW_FRAMEBUFFER,previous)
        finally:context.make_uncurrent()


class OrderedDynamicSession(PulseContextSession,OrderedDynamicPhysicalInit):
    def __init__(self,inventory,spec,output):
        if not isinstance(inventory,CollectionInventory):raise ValueError('validated fixed inventory required')
        self.inventory=inventory
        super().__init__(spec,output)

    def capture_fixed_rgb(self,output,name):
        row=NearFieldCapture.capture_fixed_rgb(self,output,name)
        camera=self.ctx.build.camera
        witness=verify_order(camera,self.raster_order)
        write_json(output/(name.replace('rgb_','raster_')+'.json'),dict(
            physical_sample_index=len(self.samples)-1,order=witness,precision=precision_readback(camera)))
        return row
