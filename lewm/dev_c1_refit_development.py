"""C1 with the stage-2 fairness-refit command model ("C1R"; development; Andrew, 3 October 2026: "C1 is refit on the
patch data as a fairness check ... It runs alongside unchanged C1").

The runtime loads the deployed command-history model in its constructor (self.command_model, self.command_fit_sha256;
lewm/short_pulse_navigation_runtime_development.py). This mixin, outermost, replaces both after construction with the refit
from scripts/fit_go2_stage2_c1_refit_development.py (`<capability root>/stage2_c1_refit_v1/command_only.npz`), checked
against the SHA-256 that fit recorded. Everything else in C1 is unchanged, and the logged command_history_fit_sha256 names
the refit.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

REFIT = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/'
             'go2_navigation_capability_v1_attempt_001/stage2_c1_refit_v1')


def load_refit():
    path = REFIT/'command_only.npz'
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != json.loads((REFIT/'result.json').read_text())['refit_sha256']:
        raise ValueError('C1 refit differs from its recorded fit')
    with np.load(path, allow_pickle=False) as arrays:
        return {k: arrays[k].copy() for k in ('mean', 'scale', 'bias', 'coefficient')}, sha


class RefitCommandModelMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, 'command_model'):
            raise ValueError('C1 runtime with a command-history model required')
        self.command_model, self.command_fit_sha256 = load_refit()
