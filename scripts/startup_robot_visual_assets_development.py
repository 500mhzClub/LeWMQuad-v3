"""Seven exact installed visual assets; no recursive asset discovery or export."""
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry,origin_transform
from lewm.causal_ground_plane_development import verify_robot_geometry
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.run_go2_successive_choice_maze_development_v1 import digest

ASSET_ROOT=URDF.parent.parent/'dae'
ASSETS={
 'base.dae': 'e52bebbb5c7ff1f6aedee620c9b5b349f6e965360ecc63a9bb743eeff2d0be64',
 'hip.dae': '8735e10617afe252cef0844c1d139d15bc6d4c653660d742fea9fe05c2704d8f',
 'thigh.dae': 'a622381ae308897ff143b8bbf02f075457c342b22984205040424a5a983246fa',
 'calf.dae': '29872fa1a435b92bc2b395fdfa7f0aee26101db3c494681896bd4ef24ee7883c',
 'foot.dae': 'ffbd95fd641866ce9bd277aa249b92dafc9148d70131fa2c52709a35ce5b710b',
 'thigh_mirror.dae': 'c7bae0b0565c2aa4b7cc15a538d795e9153e0f10c821cc18429e92693172bdc6',
 'calf_mirror.dae': '79c66be7ed06760bd5001c067a1fb0ac517f639d8b1a5f89be471903b3f359e1'}


def verify_assets():
    verify_robot_geometry(URDF)
    for name,sha in ASSETS.items():
        path=ASSET_ROOT/name
        if path.resolve()!=path or digest(path)!=sha: raise ValueError('exact installed visual asset changed')
        root=ET.parse(path).getroot()
        if root.findall('.//{*}image') or root.findall('.//{*}init_from'):
            raise ValueError('undeclared external visual dependency')
    return {str(ASSET_ROOT/n):h for n,h in ASSETS.items()}


def robot_mesh(joints,rotation,position):
    verify_assets(); geometry=ArticulatedCollisionGeometry(URDF); transforms,_=geometry.transforms(joints)
    root=ET.parse(URDF).getroot(); cache={}; pieces=[]; witnesses=[]
    for link in root.findall('link'):
        for visual in link.findall('visual'):
            source=visual.find('geometry/mesh')
            if source is None or source.get('scale') is not None: raise ValueError('declared unscaled visual mesh required')
            filename=source.get('filename'); name=filename.removeprefix('../dae/')
            if filename!='../dae/'+name or name not in ASSETS: raise ValueError('undeclared visual mesh path')
            if name not in cache:
                scene=trimesh.load_scene(ASSET_ROOT/name,file_type='dae',process=False,allow_remote=False)
                if not scene.geometry or any(not isinstance(v,trimesh.Trimesh) for v in scene.geometry.values()):
                    raise ValueError('triangle-only visual asset required')
                # Untextured COLLADA PBR materials return a single RGBA value.
                # Expand per geometry BEFORE concatenating different materials.
                for part in scene.geometry.values():
                    if part.visual.kind=='texture':
                        colors=np.asarray(part.visual.to_color().vertex_colors)
                        if colors.shape==(4,): colors=np.tile(colors,(len(part.vertices),1))
                        if colors.shape!=(len(part.vertices),4): raise ValueError('per-vertex RGBA required')
                        part.visual=trimesh.visual.ColorVisuals(mesh=part,vertex_colors=colors)
                mesh=scene.to_geometry()
                if np.any(mesh.visual.vertex_colors[:,3]!=255): raise ValueError('opaque visual material required')
                cache[name]=mesh
            mesh=cache[name].copy(); T=transforms[link.get('name')]@origin_transform(visual.find('origin'))
            mesh.apply_transform(T)
            mesh.vertices=mesh.vertices@np.asarray(rotation).T+position
            pieces.append(mesh); witnesses.append(dict(link=link.get('name'),asset=name,vertices=len(mesh.vertices),
                faces=len(mesh.faces),world_lower=mesh.bounds[0].tolist(),world_upper=mesh.bounds[1].tolist()))
    if len(pieces)!=17: raise ValueError('all17URDFvisualinstances required')
    return trimesh.util.concatenate(pieces),witnesses
