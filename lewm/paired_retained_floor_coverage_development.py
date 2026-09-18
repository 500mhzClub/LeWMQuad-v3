"""Read-only joint view of two synchronized, already measured floor histories."""
from copy import deepcopy
from lewm.retained_floor_patch_development import RetainedFloorPatches


class PairedRetainedFloorPatches:
    def __init__(self,primary,auxiliary):
        if (not isinstance(primary,RetainedFloorPatches) or not isinstance(auxiliary,RetainedFloorPatches)
                or not 0<len(primary.frames)==len(auxiliary.frames)<=256):
            raise ValueError('bounded complete paired measured histories required')
        frames=[]
        for i,(a,b) in enumerate(zip(primary.frames,auxiliary.frames,strict=True)):
            if (a['witness']['frame']!=i or b['witness']['frame']!=i
                    or a['witness']['measured_ns']!=b['witness']['measured_ns']
                    or a['floor_height']!=b['floor_height']):
                raise ValueError('same measured epoch and fixed floor required')
            for stream,frame in (('primary',a),('auxiliary',b)):
                if frame['prefix'].flags.writeable:raise ValueError('immutable measured floor prefix required')
                frames.append(frame|dict(witness=deepcopy(frame['witness'])|dict(sensor_stream=stream)))
        self.frames=tuple(frames)

    def coverage(self,centres,radius=.022):
        return RetainedFloorPatches.coverage(self,centres,radius)
