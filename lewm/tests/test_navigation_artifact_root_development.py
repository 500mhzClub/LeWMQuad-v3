"""Synthetic temporary artifact-root tests; no external data discovery."""
from pathlib import Path
import pytest
from scripts import navigation_artifact_root_development as authority


def root(monkeypatch,tmp_path):
    base=tmp_path/'new_artifacts';monkeypatch.setattr(authority,'BASE',base)
    return base/'go2_test_attempt_001'


def test_exclusive_owned_attempt_and_hash_validation(monkeypatch,tmp_path):
    out=root(monkeypatch,tmp_path);authority.create_output(out)
    (out/'data.json').write_text('{}')
    h=authority.digest(out/'data.json');authority.verify_artifacts(out,{'data.json':h})
    with pytest.raises(ValueError):authority.create_output(out)
    (out/'data.json').write_text('changed')
    with pytest.raises(ValueError):authority.verify_artifacts(out,{'data.json':h})


@pytest.mark.parametrize('name',['../escape','/absolute','sealed/data','sealed_old/data','sealed_test.json','a/../b','a//b',''])
def test_protected_and_escaping_artifact_paths_rejected(monkeypatch,tmp_path,name):
    out=root(monkeypatch,tmp_path);authority.create_output(out)
    with pytest.raises(ValueError):authority.artifact_path(out,name)


def test_symlink_leaf_and_parent_rejected(monkeypatch,tmp_path):
    out=root(monkeypatch,tmp_path);authority.create_output(out)
    (out/'real').write_text('data');(out/'link').symlink_to(out/'real')
    with pytest.raises(ValueError):authority.artifact_path(out,'link')
    target=out/'directory';target.mkdir();(target/'data').write_text('data');(out/'alias').symlink_to(target,target_is_directory=True)
    with pytest.raises(ValueError):authority.artifact_path(out,'alias/data')


def test_wrong_root_and_bad_hash_rejected(monkeypatch,tmp_path):
    out=root(monkeypatch,tmp_path);authority.create_output(out)
    with pytest.raises(ValueError):authority.validate_root(tmp_path/'go2_test_attempt_001')
    with pytest.raises(ValueError):authority.validate_root(authority.BASE/'sealed_old_attempt_001',must_exist=False)
    with pytest.raises(ValueError):authority.verify_artifacts(out,{'data':'not-sha'})
