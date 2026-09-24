"""Explicit new model identity; never load residual weights as absolute heads."""
import hashlib
import io
import torch
from lewm.nominal_motion_residual_learning_development import NominalResidualTrainer, PARAMETERIZATION
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.observation_horizon_snapshot_development import config

SCHEMA='nominal_motion_residual_snapshot.v1'


def validate_payload(payload,expected_binding,expected_config):
    if payload['schema']!=SCHEMA or payload['binding']!=expected_binding:
        raise ValueError('explicit nominal-residual snapshot identity required')
    saved=payload['trainer']
    if (saved['schema']!='nominal_motion_residual_training_development.v1'
            or saved['motion_parameterization']!=PARAMETERIZATION or saved['failed']
            or {k:saved[k] for k in expected_config}!=expected_config):
        raise ValueError('correct nonfailed residual parameterization and configuration required')
    clone=NominalResidualTrainer(**{k:v for k,v in expected_config.items() if k!='updates'})
    if clone.initial_sha256!=saved['initial_sha256']:
        raise ValueError('initial model identity changed')
    clone.model.load_state_dict(saved['model_state'],strict=True)
    if (state_digest(clone.model.state_dict())!=saved['model_sha256']
            or not all(torch.isfinite(p).all() for p in clone.model.parameters())):
        raise ValueError('finite exact model weights required')
    clone.updates=saved['updates']; clone.evaluation_only=True; clone.model.eval()
    return clone


def load_snapshot(directory,name,*,sha256,expected_binding,expected_config):
    path=directory/name
    if path.is_symlink() or path.stat().st_size>64*1024**2:
        raise ValueError('bounded regular snapshot required')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha256:
        raise ValueError('checkpoint bytes changed')
    return validate_payload(torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True),
                            expected_binding,expected_config)
