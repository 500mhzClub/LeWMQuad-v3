"""Resolve collision shapes against an observed robot-link roster, not hints."""
import xml.etree.ElementTree as ET

from lewm.causal_ground_plane_development import verify_robot_geometry


def resolve_native_groups(urdf_path, shape_rows, native_robot_link_names):
    verify_robot_geometry(urdf_path)
    native = set(native_robot_link_names)
    root = ET.parse(urdf_path).getroot()
    links = {e.get('name') for e in root.findall('link')}
    if not native or not native <= links or 'base' not in native:
        raise ValueError('explicit native ROBOT link roster within calibrated URDF required')
    parents = {j.find('child').get('link'): (j.find('parent').get('link'), j.get('type')) for j in root.findall('joint')}
    result = {}
    for shape in shape_rows:
        link = shape['link']
        seen = set()
        while link not in native:
            if link in seen or link not in parents:
                raise ValueError('unresolved native collision ancestor')
            seen.add(link)
            parent, kind = parents[link]
            if kind != 'fixed':
                raise ValueError('cannot collapse a missing movable joint into its parent')
            link = parent
        if shape['shape_id'] in result:
            raise ValueError('duplicate collision shape identity')
        result[shape['shape_id']] = link
    return result
