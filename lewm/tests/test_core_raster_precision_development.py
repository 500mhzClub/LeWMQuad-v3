from types import SimpleNamespace as NS
import sys
import pytest
from lewm_genesis.core_raster_precision_development import precision_readback

@pytest.mark.parametrize('fault',[None,'attachment_error','zero_bits','zero_samples','bad_position','wrong_target'])
def test_attachment_query_and_cleanup(monkeypatch,fault):
    calls=[];state={'fb':11}
    constants=['GL_DRAW_FRAMEBUFFER','GL_DRAW_FRAMEBUFFER_BINDING','GL_SUBPIXEL_BITS','GL_DEPTH_ATTACHMENT',
        'GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE','GL_SAMPLES','GL_SAMPLE_POSITION']
    gl=NS(**{k:k for k in constants})
    def get(k):
        assert k!='GL_DEPTH_BITS'
        return {'GL_DRAW_FRAMEBUFFER_BINDING':99 if fault=='wrong_target' else state['fb'],
                'GL_SUBPIXEL_BITS':8,'GL_SAMPLES':0 if fault=='zero_samples' else 4}[k]
    def attachment(target,which,pname):
        calls.append(('attachment',target,which,pname))
        assert (target,which,pname)==('GL_DRAW_FRAMEBUFFER','GL_DEPTH_ATTACHMENT','GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE')
        if fault=='attachment_error':raise RuntimeError('readback failure')
        return 0 if fault=='zero_bits' else 24
    def bind(target,fb):state['fb']=fb
    gl.glGetIntegerv=get;gl.glBindFramebuffer=bind;gl.glGetFramebufferAttachmentParameteriv=attachment
    gl.glGetMultisamplefv=lambda _,i:[2,.5] if fault=='bad_position' else [.25,.75]
    monkeypatch.setitem(sys.modules,'OpenGL.GL',gl)
    context=NS(make_current=lambda:calls.append('current'),make_uncurrent=lambda:calls.append('uncurrent'))
    camera=NS(uid=1,_rasterizer=NS(_renderer=context,_camera_targets={1:NS(_main_fb=11,_main_fb_ms=12)}))
    if fault:
        with pytest.raises((ValueError,RuntimeError)):precision_readback(camera)
    else:
        result=precision_readback(camera)
        assert result['depth_target_depth_bits']==24 and result['rgb_target_samples']==4
    assert calls[0]=='current' and calls[-1]=='uncurrent'
    assert state['fb']==(99 if fault=='wrong_target' else 11)
