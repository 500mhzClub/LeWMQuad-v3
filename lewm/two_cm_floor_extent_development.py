"""Explicit process-local development treatment of floor-normal observability."""
from lewm import joint_measured_floor_plane_development as plane


def configure():
    # Call before constructing any observer in a fresh process. All plane
    # composition and readout functions then share this explicit treatment.
    plane.MINIMUM_SECOND_EXTENT_M = .02


def initialize_pose():
    from lewm.joint_camera_anchor_tracker_development import initialize_joint_camera_pose
    configure(); initialize_joint_camera_pose()


def initialize_registration():
    from lewm.partial_height_round_trip_development import initialize_registration as original
    configure(); original()


def initialize_mapping():
    from lewm.fine_stored_obstacle_routing_development import initialize_fine_mapping
    configure(); initialize_fine_mapping()


def initialize_obstacles():
    from lewm.partial_height_round_trip_development import initialize_obstacles as original
    configure(); original()
