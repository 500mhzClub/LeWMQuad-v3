"""Only identical single-link files on one volume can share proposed storage."""
from copy import deepcopy
import pytest
from scripts.inventory_go2_training_artifact_duplicates_v1 import duplicate_groups


def records():
    return [dict(root='root',path=str(i),sha256='a'*64,metadata=dict(
        dev=1,ino=i,mode=33188,uid=1000,gid=1000,size=500,nlink=1,blocks=2+i)) for i in range(3)]


def test_preserves_largest_allocation_and_lists_every_other_path():
    rows=records();before=deepcopy(rows);groups=duplicate_groups(rows)
    assert rows==before and len(groups)==1
    assert groups[0]['canonical']['path']=='2'
    assert {r['path'] for r in groups[0]['duplicate_paths']}=={'0','1'}
    assert groups[0]['potential_allocated_saving_bytes']==(2+3)*512


@pytest.mark.parametrize('field',['dev','mode','uid','gid','size','nlink','sha256'])
def test_incompatible_files_never_share_storage(field):
    rows=records()[:2]
    if field=='sha256':rows[1][field]='b'*64
    else:rows[1]['metadata'][field]+=1
    assert duplicate_groups(rows)==[]


def test_repeated_path_or_contradictory_inode_rejects():
    rows=records()[:2];rows[1]['path']=rows[0]['path']
    with pytest.raises(ValueError):duplicate_groups(rows)
    rows=records()[:2];rows[1]['metadata']['ino']=rows[0]['metadata']['ino']
    with pytest.raises(ValueError):duplicate_groups(rows)
