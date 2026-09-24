"""Read identity from a camera's existing context without changing render state."""


def renderer_identity_readback(camera):
    from OpenGL.GL import (glGetString, glGetIntegerv, GL_VENDOR, GL_RENDERER,
        GL_VERSION, GL_SHADING_LANGUAGE_VERSION, GL_CONTEXT_PROFILE_MASK,
        GL_DRAW_FRAMEBUFFER_BINDING)
    raster = camera._rasterizer; context = raster._renderer
    target = raster._camera_targets[camera.uid]
    context.make_current()
    try:
        if int(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING)) != int(target._main_fb):
            raise ValueError('the camera depth framebuffer must be current')
        values = {}
        for key, enum in (('vendor', GL_VENDOR), ('renderer', GL_RENDERER),
                ('version', GL_VERSION), ('shading_language_version', GL_SHADING_LANGUAGE_VERSION)):
            raw = glGetString(enum)
            if not isinstance(raw, bytes) or not 1 <= len(raw) <= 4096:
                raise ValueError('bounded nonempty current-context identity string required')
            values[key] = raw.decode('ascii')
        profile = int(glGetIntegerv(GL_CONTEXT_PROFILE_MASK))
        if not 0 <= profile <= 0xffff: raise ValueError('bounded actual profile mask required')
        device = getattr(context._platform, '_egl_device', None)
        name = None if device is None else device.name
        if name is not None and (not isinstance(name, str) or not 1 <= len(name) <= 4096):
            raise ValueError('bounded optional EGL device identity required')
        return dict(**values, profile_mask=profile, egl_device_name=name,
            platform_class=type(context._platform).__module__+'.'+type(context._platform).__name__,
            renderer_declares_software=bool(context._is_software),
            camera_uid=str(camera.uid), framebuffer_matches_camera_depth_target=True,
            scope='queried current camera context; no inference about other contexts or historical frames',
            source_implementation_equivalence_proven=False, raster_error_bound_proven=False)
    finally:
        context.make_uncurrent()
