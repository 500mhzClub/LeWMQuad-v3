"""Record context at existing captures; public packets and renders are inherited."""
from scripts.dual_camera_novel_maze_session_development import DualCameraNovelMazeSession
from scripts.maze_renderer_witness_development import capture_witness, validate_pair, PHASES
from scripts.run_go2_successive_choice_maze_development_v1 import write_json

ARTIFACT = 'renderer_capture_witnesses.json'


class RendererWitnessDualCameraMazeSession(DualCameraNovelMazeSession):
    def __init__(self, *args, **kwargs):
        self.renderer_witnesses = dict(primary=[], paired=[], failures=[])
        super().__init__(*args, **kwargs)

    def _require_live_witness(self):
        if self.renderer_witnesses['failures']:
            raise ValueError('renderer witness failure is terminal; no reacquisition')

    def _failed(self, phase, frame, error):
        if not self.renderer_witnesses['failures']:
            self.renderer_witnesses['failures'].append(dict(phase=phase, frame=frame, reason=repr(error)))

    def capture_fixed_rgb(self, output, name):
        self._require_live_witness(); frame = len(self.depth_manifest)
        try:
            if name != f'rgb_{frame:04d}' or len(self.renderer_witnesses['primary']) != frame:
                raise ValueError('consecutive original primary acquisition required')
            original = super().capture_fixed_rgb(output, name)
            row = capture_witness(self, frame=frame, phase=PHASES[0], pixel_hashes=dict(
                primary_rgb_sha256=original['rgb_sha256'],
                primary_native_depth_sha256=self.depth_audit[frame]['native_depth_sha256']))
            if self.renderer_witnesses['primary'] and row['context'] != self.renderer_witnesses['primary'][0]['context']:
                raise ValueError('renderer context changed within the episode')
            self.renderer_witnesses['primary'].append(row)
            return original
        except Exception as error:
            self._failed(PHASES[0], frame, error); raise

    def sensor_packets(self):
        self._require_live_witness(); frame = (len(self.samples)-750)//50
        try:
            original = super().sensor_packets()
            policy, depth, fast, auxiliary, image, now = original
            index = len(self.model_manifest)-1
            if index != frame or now != 1_500_000_000+100_000_000*frame:
                raise ValueError('same paired acquisition boundary required')
            records = self.renderer_witnesses['paired']
            if len(records) == frame:
                a = self.auxiliary_audit[frame]
                row = capture_witness(self, frame=frame, phase=PHASES[1], pixel_hashes=dict(
                    primary_rgb_sha256=image['primary_rgb_sha256'],
                    primary_native_depth_sha256=self.depth_audit[frame]['native_depth_sha256'],
                    auxiliary_rgb_sha256=a['rgb_sha256'], auxiliary_native_depth_sha256=a['native_depth_sha256']))
                validate_pair(self.renderer_witnesses['primary'][frame], row)
                records.append(row)
            if len(records) != frame+1:
                raise ValueError('one renderer witness per original paired acquisition required')
            return original
        except Exception as error:
            self._failed(PHASES[1], frame, error); raise

    def persist_observations(self, output):
        try:
            super().persist_observations(output)
        finally:
            write_json(output/ARTIFACT, self.renderer_witnesses)
