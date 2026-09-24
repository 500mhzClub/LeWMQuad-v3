"""Core-profile framebuffer attachment query; no deprecated GL_DEPTH_BITS query."""
import numpy as np

def precision_readback(camera):
    from OpenGL.GL import (glGetIntegerv,glBindFramebuffer,glGetMultisamplefv,glGetFramebufferAttachmentParameteriv,
        GL_DRAW_FRAMEBUFFER,GL_DRAW_FRAMEBUFFER_BINDING,GL_SUBPIXEL_BITS,
        GL_DEPTH_ATTACHMENT,GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE,GL_SAMPLES,GL_SAMPLE_POSITION)
    raster=camera._rasterizer;context=raster._renderer;target=raster._camera_targets[camera.uid]
    previous=None;context.make_current()
    try:
        previous=int(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING))
        if previous!=int(target._main_fb):raise ValueError('depth single-sample framebuffer must be current')
        bits=int(glGetIntegerv(GL_SUBPIXEL_BITS));depth_bits=int(glGetFramebufferAttachmentParameteriv(
            GL_DRAW_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE))
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

