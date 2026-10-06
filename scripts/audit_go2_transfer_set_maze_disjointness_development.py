"""Maze-level disjointness of the 240-window transfer set (Andrew's question, 29 Sep 2026).

Compares the two transfer mazes with the scene of every recording directory used to train
C3-v2's readout and C4-v2 (the matched training data), and, separately, with the recordings
behind C3's frozen action-conditioned predictor. Three identities are compared:
exact wall geometry, the canonical abstract topology, and the canonical metric layout
(invariant to translation, reflection and right-angle rotation). Reads specifications only.
"""
import hashlib
import json
from pathlib import Path

from lewm import decision_headroom_json_v42_development as output
from lewm.independent_layout_identity_development import metric_identity, topology_identity
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner

BASE = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
TRANSFER = BASE.parent/'go2_maze_view_transfer_v1_attempt_001'
PREDICTOR = owner.REPO/'.generated/navigation_development_artifacts_v1/go2_horizon_dense_predictor_v1_attempt_001'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(directory):
    spec = json.loads((Path(directory)/'specification.json').read_text())
    walls = sorted((tuple(round(v, 6) for v in w['centre_xyz']), tuple(round(v, 6) for v in w['size_xyz']), round(w['yaw_rad'], 6))
                   for w in spec['geometry']['wall_boxes'])
    row = dict(scene_id=spec.get('scene_id'), family=spec.get('family'), walls=len(walls),
               wall_geometry_sha256=hashlib.sha256(json.dumps(walls).encode()).hexdigest(), topology_sha256=None, metric_sha256=None)
    layout = spec.get('evaluation_layout')
    if layout:
        edges = [tuple(tuple(p) for p in e) for e in layout['edges']]
        row['topology_sha256'] = topology_identity(edges)['sha256']
        row['metric_sha256'] = metric_identity(edges, pitch_m=layout['pitch_m'], wall_thickness_m=.08, wall_height_m=1.4)['sha256']
    return row


def compare(name, directories, transfer):
    rows = {d: identity(d) for d in sorted(directories)}
    keys = ('wall_geometry_sha256', 'topology_sha256', 'metric_sha256')
    shared = {k: sorted({r[k] for r in rows.values() if r[k]} & {t[k] for t in transfer.values() if t[k]}) for k in keys}
    return dict(source=name, directories=len(rows), scenes=len({r['wall_geometry_sha256'] for r in rows.values()}),
                maze_scenes=len({r['metric_sha256'] for r in rows.values() if r['metric_sha256']}),
                families=sorted({r['family'] for r in rows.values()}), wall_counts=sorted({r['walls'] for r in rows.values()}),
                shared_with_transfer=shared, disjoint=not any(shared.values()))


def main():
    output.install(BASE)
    transfer = {str(d): identity(d) for d in sorted(TRANSFER.glob('case_*')) if d.is_dir()}
    train = json.loads((BASE/'c3v2_data_v1/train_samples.json').read_text())
    by_root = {}
    for row in train:
        by_root.setdefault(Path(row['directory']).parent.name, set()).add(row['directory'])
    predictor_paths = json.loads((PREDICTOR/'frame_paths.json').read_text())
    predictor_dirs = {str(Path(p).parent) for p in predictor_paths if (Path(p).parent/'specification.json').exists()}
    sources = [compare(name, dirs, transfer) for name, dirs in sorted(by_root.items())]
    predictor = compare('C3 frozen predictor training recordings', predictor_dirs, transfer)
    result = dict(schema='transfer_set_maze_disjointness.v1',
        transfer=dict(cases=len(transfer), mazes=len({t['metric_sha256'] for t in transfer.values()}),
                      metric_sha256=sorted({t['metric_sha256'] for t in transfer.values()})),
        matched_training_data=sources, matched_training_data_disjoint=all(s['disjoint'] for s in sources),
        c3_frozen_predictor=predictor | dict(frame_paths=len(predictor_paths),
                                             frame_paths_without_specification=sum(not (Path(p).parent/'specification.json').exists() for p in predictor_paths)),
        inputs_sha256={'train_samples.json': sha(BASE/'c3v2_data_v1/train_samples.json'), 'predictor_frame_paths.json': sha(PREDICTOR/'frame_paths.json')},
        auditor_sha256=sha(__file__))
    owner.save(BASE/'c3v2_data_v1/transfer_set_maze_disjointness.json', result)
    print(json.dumps(dict(transfer=result['transfer'], matched_training_data_disjoint=result['matched_training_data_disjoint'],
                          sources=[{k: s[k] for k in ('source', 'directories', 'scenes', 'maze_scenes', 'families', 'wall_counts', 'disjoint')} for s in sources],
                          predictor={k: predictor[k] for k in ('directories', 'scenes', 'maze_scenes', 'families', 'disjoint')} | dict(
                              without_spec=result['c3_frozen_predictor']['frame_paths_without_specification'])), indent=1))


if __name__ == '__main__':
    main()
