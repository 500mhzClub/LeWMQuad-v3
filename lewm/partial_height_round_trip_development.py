"""Integrate explicit partial-height observations into the continuous development runtime."""
from lewm.eligible_floor_registration_development import bind
from lewm.partial_floor_height_development import PartialHeightRegistration,read_pose,fit_gyro_height
from lewm import process_registered_round_trip_development as registration_process
from lewm import process_mapped_runtime_development as mapping_process
from lewm import independent_depth_process_development as depth_process
from lewm.multirate_routing_map_development import MultirateRoutingMap
from lewm.independent_depth_obstacle_development import IndependentDepthObstacles
from lewm.fine_obstacle_round_trip_development import FineDepthObstacles,FineObstacleRoundTripRuntime


class PartialHeightMap(MultirateRoutingMap):
    update=bind(MultirateRoutingMap.update,current_measured_floor_pose=read_pose)


_partial_observe=bind(IndependentDepthObstacles._observe,CELL_M=.01,fit_joint_plane=fit_gyro_height)


class PartialHeightObstacles(FineDepthObstacles):
    def _observe(self,policy,depth,fast,auxiliary,now):
        if self.frames==0:return FineDepthObstacles._observe(self,policy,depth,fast,auxiliary,now)
        return _partial_observe(self,policy,depth,fast,auxiliary,now)


def initialize_registration():
    registration_process.initialize_registration()
    registration_process._registration=PartialHeightRegistration()


def initialize_mapping():
    mapping_process.initialize_mapping()
    mapping_process._mapper=PartialHeightMap()


def initialize_obstacles():
    depth_process.initialize_obstacles()
    depth_process._observer=PartialHeightObstacles()


class PartialHeightRoundTripRuntime(FineObstacleRoundTripRuntime):
    _pose=staticmethod(read_pose)
