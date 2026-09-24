"""Instance-local floor/wall/robot draw order for sensor characterization."""
from types import MethodType,SimpleNamespace
from lewm_genesis.ordered_union_raster_development import surface_nodes


def install_order(build):
    context=build.camera._rasterizer._context;scene=context._scene
    static=[context.rigid_nodes[g.uid] for e in build.visual_surfaces for g in e.vgeoms]
    robots=[context.rigid_nodes[g.uid] for g in build.robot.vgeoms]
    roles,witnesses=surface_nodes(SimpleNamespace(mesh_nodes=static,get_pose=scene.get_pose))
    ordered=[roles['floor'],roles['walls'],*robots]
    if not robots or len(set(ordered))!=len(ordered) or set(ordered)!=set(scene.mesh_nodes):
        raise ValueError('complete distinct floor/wall/visible-robot population required')
    def fixed_sort(this):
        if set(this.mesh_nodes)!=set(ordered):raise ValueError('visible population changed')
        return list(ordered)
    scene.sorted_mesh_nodes=MethodType(fixed_sort,scene);scene._meshes_updated=True
    return dict(order='floor_walls_robot_visual_geometry_order',surfaces=witnesses,
        robot_visual_geometries=len(robots),total_nodes=len(ordered))


def verify_order(build,expected):
    context=build.camera._rasterizer._context;scene=context._scene
    static=[context.rigid_nodes[g.uid] for e in build.visual_surfaces for g in e.vgeoms]
    roles,witnesses=surface_nodes(SimpleNamespace(mesh_nodes=static,get_pose=scene.get_pose))
    robots=[context.rigid_nodes[g.uid] for g in build.robot.vgeoms]
    wanted=[roles['floor'],roles['walls'],*robots]
    if (witnesses!=expected['surfaces'] or len(robots)!=expected['robot_visual_geometries']
            or set(scene.mesh_nodes)!=set(wanted) or list(context.jit.node_list)!=wanted
            or list(scene.sorted_mesh_nodes())!=wanted):raise ValueError('actual visible-robot draw order changed')
    return expected
