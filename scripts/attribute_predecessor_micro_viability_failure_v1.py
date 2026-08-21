#!/usr/bin/env python3
"""Read-only attribution of the completed direct-nonviability predecessor."""
from __future__ import annotations
import json, math, os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lewm.safety import lightweight_one_tick_viability_model_v1 as CORE

LEDGER=Path.home()/".cache/lewm_go2_temporal_v03/development_micro_viability_model_screen_v1/row_level_model_evidence_v1.jsonl"
EXPECTED="555ba6d2678e543cf78d6a53977eceeaa5bddf60a6c16c2510ee028db9f7cba2"
OUT=ROOT/".generated/two_ply_set_structured_micro_viability_v1/predecessor_failure_attribution.json"

def metrics(rows,kind):
    if kind=="contact": labels=np.asarray([r["contact"] for r in rows],bool);scores=np.asarray([r["contact_probability"] for r in rows])
    else:
        rows=[r for r in rows if not r["contact"]];labels=np.asarray([r["n_safe"]==0 for r in rows],bool);scores=np.asarray([r["nonviability_probability"] for r in rows])
    def dist(mask):
        values=scores[mask]
        return {"count":len(values),"mean":float(values.mean()) if len(values) else None,"median":float(np.median(values)) if len(values) else None,
                "p10":float(np.percentile(values,10)) if len(values) else None,"p90":float(np.percentile(values,90)) if len(values) else None}
    auc=CORE.auc(labels,scores);ap=CORE.average_precision(labels,scores)
    return {"auc":None if math.isnan(auc) else auc,"ap":None if math.isnan(ap) else ap,"positive":dist(labels),"negative":dist(~labels)}

def main():
    import hashlib
    if hashlib.sha256(LEDGER.read_bytes()).hexdigest()!=EXPECTED:raise RuntimeError("predecessor ledger binding failed")
    rows=[json.loads(line) for line in LEDGER.read_text().splitlines()]
    splits={name:[r for r in rows if r["split"]==name] for name in ("training","calibration","heldout")}
    result={"schema":"two_ply_predecessor_failure_attribution_v1","ledger_sha256":EXPECTED,"splits":{}}
    for name,part in splits.items():
        result["splits"][name]={"rows":len(part),"contact":metrics(part,"contact"),"nonviability":metrics(part,"nonviability"),
            "per_family":{family:{"contact":metrics([r for r in part if r["family"]==family],"contact"),
                                  "nonviability":metrics([r for r in part if r["family"]==family],"nonviability")} for family in CORE.FAMILIES},
            "per_candidate":{str(index):{"contact":metrics([r for r in part if r["action_index"]==index],"contact"),
                                           "nonviability":metrics([r for r in part if r["action_index"]==index],"nonviability")} for index in range(14)}}
    train=result["splits"]["training"];held=result["splits"]["heldout"]
    result["fit_to_heldout_degradation"]={"contact_auc":train["contact"]["auc"]-held["contact"]["auc"],
        "contact_ap":train["contact"]["ap"]-held["contact"]["ap"],"nonviability_auc":train["nonviability"]["auc"]-held["nonviability"]["auc"],
        "nonviability_ap":train["nonviability"]["ap"]-held["nonviability"]["ap"]}
    result["classifications"]=["FIT_TO_HELDOUT_GENERALISATION_FAILURE","DIRECT_NONVIABILITY_TARGET_MISALIGNMENT"]
    result["model_underfit_supported"]=False;result["content_digest"]=CORE.digest(result)
    OUT.parent.mkdir(parents=True,exist_ok=True);tmp=OUT.with_suffix(".tmp");tmp.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=True)+"\n");os.replace(tmp,OUT)
    print(json.dumps({"splits":{k:{"contact":v["contact"],"nonviability":v["nonviability"]} for k,v in result["splits"].items()},
                      "degradation":result["fit_to_heldout_degradation"],"classifications":result["classifications"]},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
