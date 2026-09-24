"""Same mounted capture and motion; only core-profile precision query differs."""
from lewm_genesis.core_raster_precision_development import precision_readback
from lewm_genesis.ordered_union_raster_development import verify_order
from scripts.ordered_dynamic_session_development import OrderedDynamicSession
from scripts.near_field_rgbd_capture_development import NearFieldCapture
from scripts.run_go2_successive_choice_maze_development_v1 import write_json


class CoreOrderedDynamicSession(OrderedDynamicSession):
    def capture_fixed_rgb(self,output,name):
        row=NearFieldCapture.capture_fixed_rgb(self,output,name)
        camera=self.ctx.build.camera
        witness=verify_order(camera,self.raster_order)
        write_json(output/(name.replace('rgb_','raster_')+'.json'),dict(
            physical_sample_index=len(self.samples)-1,order=witness,precision=precision_readback(camera)))
        return row
