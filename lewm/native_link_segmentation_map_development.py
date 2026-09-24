"""Validate the renderer's explicit background sentinel and link identities."""
import numpy as np


def decode(mapping,robot_entity_idx):
    if not isinstance(mapping,dict) or mapping.get(0)!=-1:
        raise ValueError('exact native background sentinel 0 -> -1 required')
    if isinstance(robot_entity_idx,bool) or not isinstance(robot_entity_idx,(int,np.integer)) or robot_entity_idx<0:
        raise ValueError('nonnegative native robot entity identity required')
    clean={};robot_ids=[]
    for key,value in mapping.items():
        if isinstance(key,bool) or not isinstance(key,(int,np.integer)) or key<0:
            raise ValueError('nonnegative integer segmentation index required')
        if key==0:
            clean['0']=-1;continue
        if (not isinstance(value,tuple) or len(value)!=2 or any(isinstance(v,bool)
                or not isinstance(v,(int,np.integer)) or v<0 for v in value)):
            raise ValueError('exact nonbackground entity/link identity required')
        clean[str(int(key))]=list(map(int,value))
        if value[0]==robot_entity_idx:robot_ids.append(int(key))
    if not robot_ids:raise ValueError('native robot visual link identities required')
    return clean,robot_ids
