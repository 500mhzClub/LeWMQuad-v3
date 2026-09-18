"""Instance-local renderer readback variants with unchanged numerical bodies."""
import numba as nb


def readback_variants(camera):
    return renderer_variants(camera, ('_read_depth_buf', '_read_color_buf'))


def renderer_variants(camera, names):
    jit = camera._rasterizer._camera_targets[camera.uid].jit
    originals, replacements, receipts = {}, {}, []
    for name in names:
        if name not in ('_forward_pass', '_read_depth_buf', '_read_color_buf'):
            raise ValueError('explicit bounded renderer function required')
        original = getattr(jit, name)
        if original.targetoptions.get('nogil', False) or len(original.nopython_signatures) != 1:
            raise ValueError('one original non-nogil native readback signature required')
        replacement = nb.njit(original.nopython_signatures[0], nogil=True, cache=False)(original.py_func)
        originals[name], replacements[name] = original, replacement
        receipts.append(dict(function=name, original_options=original.targetoptions,
            replacement_options=replacement.targetoptions,
            same_python_body=replacement.py_func is original.py_func,
            signature=str(original.nopython_signatures[0])))
    return jit, originals, replacements, receipts


def select_readbacks(jit, functions):
    for name, function in functions.items():
        setattr(jit, name, function)
