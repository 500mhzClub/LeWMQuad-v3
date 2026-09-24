"""Release the interpreter lock during unchanged native camera drawing."""
import hashlib
import inspect
import json
from pathlib import Path

from scripts.run_go2_persistent_visual_learning_comparison_development import FreshCameraSession
from lewm_genesis.nogil_readback_development import renderer_variants, select_readbacks


def hashes(images):
    return [[hashlib.sha256(a.tobytes()).hexdigest() for a in pair] for pair in images]


class NogilDrawingMixin:
    def settle_recorded(self):
        super().settle_recorded()
        before_physics = (int(self.ctx.runner._sim_time_ns), len(self.samples))
        original_pixels, transforms = self._render_pair()
        jit, originals, replacements, treatment = renderer_variants(self.ctx.build.camera, ('_forward_pass',))
        select_readbacks(jit, replacements)
        changed_pixels, changed_transforms = self._render_pair()
        if hashes(original_pixels) != hashes(changed_pixels) or transforms != changed_transforms:
            raise ValueError('drawing lock release must preserve the fixed-pose pixels')
        if before_physics != (int(self.ctx.runner._sim_time_ns), len(self.samples)):
            raise ValueError('renderer preparation cannot advance physics')
        library = Path(inspect.getsourcefile(originals['_forward_pass'].py_func))
        receipt = dict(treatment=treatment, fixed_pose_pixels_identical=True,
            pixel_sha256=hashes(original_pixels), physics_unchanged=True,
            compiled_before_timed_execution=True, native_context_stays_on_scene_thread=True,
            original_library_path=str(library), original_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            only_drawing_lock_release_changed=True, deployment_timing_qualified=False)
        with (self.output/'renderer_scheduling_treatment.json').open('x') as stream:
            json.dump(receipt, stream, indent=2); stream.write('\n')


class NogilDrawingCameraSession(NogilDrawingMixin, FreshCameraSession):
    pass
