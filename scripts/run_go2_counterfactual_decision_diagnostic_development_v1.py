#!/usr/bin/env python3
"""Apply the declared local-intent rule to audited saved counterfactual predictions."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lewm.counterfactual_decision_diagnostic_development import decision_rows
from scripts.run_go2_rgb_body_learning_comparison_development_v1 import digest,materialize,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir',type=Path,required=True)
    output=parser.parse_args().output_dir.absolute()
    if output!=ROOT/'.generated/go2_counterfactual_decision_diagnostic_development_v1_attempt_001' or output.exists():
        raise ValueError('fresh exact secondary-analysis output required')
    learning=ROOT/'.generated/go2_rgb_body_learning_comparison_development_v1_attempt_001'
    dataset=ROOT/'.generated/go2_counterfactual_maze_dataset_development_v2_recovery_attempt_001'
    report=json.loads((learning/'result.json').read_text()); audit=json.loads((learning/'prediction_artifact_audit.json').read_text())
    if report['status']!='COMPLETE' or audit['status']!='PASS' or audit['study_result_sha256']!=digest(learning/'result.json'):
        raise ValueError('audited complete learning result required')
    source_paths=('scripts/run_go2_counterfactual_decision_diagnostic_development_v1.py',
        'lewm/counterfactual_decision_diagnostic_development.py',
        'docs/go2_counterfactual_decision_diagnostic_development_v1_2026-09-05.md')
    launch={'schema':'counterfactual_decision_diagnostic_development.v1','learning_result_sha256':digest(learning/'result.json'),
        'learning_audit_sha256':digest(learning/'prediction_artifact_audit.json'),
        'dataset_result_sha256':digest(dataset/'result.json'),
        'source_sha256':{p:digest(ROOT/p) for p in source_paths},'scope':'specified secondary offline executed-branch choices; no new physics or fitting'}
    output.mkdir(); write_json(output/'launch.json',launch)
    data=materialize(dataset,'validation')
    if data['metadata']!=report['validation_order']: raise ValueError('prediction/target ordering changed')
    rows=[]
    for model in report['models']:
        path=learning/f'{model["seed"]}-{model["condition"]}'/'validation_predictions.npz'
        if digest(path)!=model['artifact_sha256']['validation_predictions.npz']: raise ValueError('prediction bytes changed')
        with np.load(path,allow_pickle=False) as predictions:
            for name in sorted(predictions.files):
                control,head=name.split('__')
                diagnostic=decision_rows(predictions[name],data['targets'],data['metadata'])
                if diagnostic['excluded_incomplete_layouts'] or len(diagnostic['layouts'])!=8: raise ValueError('incomplete decision population')
                rows.append({'seed':model['seed'],'condition':model['condition'],'head':head,'control':control,'diagnostic':diagnostic})
    path=learning/'baseline_predictions.npz'
    if digest(path)!=report['baseline_predictions_sha256']: raise ValueError('baseline prediction bytes changed')
    baselines={}
    with np.load(path,allow_pickle=False) as predictions:
        for name in predictions.files: baselines[name]=decision_rows(predictions[name],data['targets'],data['metadata'])
    stop=np.zeros((40,8,5)); stop[...,3]=1; stop[...,4]=-30
    baselines['always_stop']=decision_rows(stop,data['targets'],data['metadata'])
    summary={}
    for key in sorted({(r['condition'],r['head'],r['control']) for r in rows}):
        selected=[r['diagnostic'] for r in rows if (r['condition'],r['head'],r['control'])==key]
        if len(selected)!=3: raise ValueError('missing training seed')
        summary['/'.join(key)]={metric:float(np.mean([r['layout_macro'][metric] for r in selected]))
            for metric in selected[0]['layout_macro']}
    if any(digest(ROOT/p)!=expected for p,expected in launch['source_sha256'].items()): raise ValueError('analysis source changed')
    result={'status':'COMPLETE','independent_development_layouts':8,'intents_per_layout':3,'model_seed_head_control_rows':rows,
        'seed_and_layout_macro':summary,'baselines':baselines,'launch_sha256':digest(output/'launch.json'),
        'scope':launch['scope']}
    write_json(output/'result.json',result)
    print(json.dumps({'status':'COMPLETE','intact':{k:v for k,v in summary.items() if k.endswith('/intact')},
        'baselines':{k:v['layout_macro'] for k,v in baselines.items()}},indent=2))


if __name__=='__main__': main()
