"""Native contact completeness guard; no simulation, repair, command or retry.

Genesis checks errno periodically. This challenge additionally checks the live
solver before every recorded sample and at termination. A capacity-compliant
array is not sufficient evidence if the solver previously overflowed.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np

from lewm.independent_tracking_challenge_development import MAX_PHYSICS_SAMPLES
from lewm.independent_tracking_recording_budget_development import MAX_CONTACT_ROWS_PER_SAMPLE

NATIVE_ROOT=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/lib/python3.12/site-packages/genesis')
NATIVE_FILES={
    '__init__.py':'35a956c8be7dc836063d2848005896bb7017c29bde758292f6415f047bad395c',
    'options/solvers.py':'2e43097685009dd8ddd7dae926ba37be6659ebb93663095944e08edb4489aa47',
    'engine/simulator.py':'9f474f0082feb1601bce629808032800d6689f65042ba4b45452846eefdc5ccc',
    'engine/solvers/rigid/rigid_solver.py':'4f16b91c2cb417916e2cebb9e2ba32cffbb6f1f096b6eac32ce08d4df3dea76e',
    'engine/solvers/rigid/collider/collider.py':'3d5863403a98cd738134116cc4768a68eb122c39f11d4d90c25329e60f37a0b9',
    'engine/solvers/rigid/collider/contact.py':'f9a9781f5959e384066f82bfff1e93a61a53eed825706dbc59071c82317bd52f',
    'engine/entities/rigid_entity/rigid_entity.py':'05c16043c05acb129f2e54723bc2d619686a1e00cfe2f6419b0e68521c940649',
}


def require(condition,message):
    if not condition:raise ValueError(message)


def native_source_bindings():
    """Seven exact installed source files only; no recursive source discovery."""
    require(NATIVE_ROOT.resolve()==NATIVE_ROOT,'exact nonsymlink installed native root')
    result={}
    for name,expected in NATIVE_FILES.items():
        path=NATIVE_ROOT/name
        require(not any(p in ('sealed','sealed_test.json') or p.startswith('sealed_') for p in path.parts),
                'protected native-source path forbidden')
        require(path.resolve()==path and path.is_file(),'ordinary exact native source required')
        with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        require(actual==expected,'native contact implementation changed: '+name)
        result[str(path)]=actual
    return result


def integer(value):
    a=np.asarray(value)
    require(a.shape==() and a.dtype.kind in 'iu','native scalar integer required')
    return int(a)


def validate_contract(row):
    fixed=dict(backend='cpu',n_envs=1,requires_grad=False,box_box_detection=False,
        enable_collision=True,enable_self_collision=True,float_dtype=np.dtype('float32').str,
        int_dtype=np.dtype('int32').str,max_collision_pairs_option=150,contacts_per_pair=5)
    require(type(row) is dict and set(row)==set(fixed)|{
        'possible_collision_pairs','allocated_collision_pairs','allocated_contacts'},'exact native capacity witness')
    require(all(type(row[k]) is type(v) and row[k]==v for k,v in fixed.items()),
            'fixed effective CPU native contact options and dtypes required')
    p=row['possible_collision_pairs'];pairs=row['allocated_collision_pairs'];contacts=row['allocated_contacts']
    require(all(type(v) is int for v in (p,pairs,contacts)) and 0<p<2**31
        and pairs==min(p,150) and contacts==pairs*5 and contacts<=MAX_CONTACT_ROWS_PER_SAMPLE,
        'actual native contact allocation differs from reviewed envelope')
    return row


def read_contract(solver,runtime):
    info=solver.collider._collider_info
    row=dict(backend='cpu' if runtime.backend==runtime.cpu else 'not_cpu',
        n_envs=integer(solver.n_envs),requires_grad=solver._static_rigid_sim_config.requires_grad,
        box_box_detection=solver._options.box_box_detection,
        enable_collision=solver._options.enable_collision,enable_self_collision=solver._options.enable_self_collision,
        float_dtype=np.dtype(runtime.np_float).str,int_dtype=np.dtype(runtime.np_int).str,
        max_collision_pairs_option=integer(solver.max_collision_pairs),
        contacts_per_pair=integer(solver.collider._collider_static_config.n_contacts_per_pair),
        possible_collision_pairs=integer(info.max_possible_pairs[None]),
        allocated_collision_pairs=integer(info.max_collision_pairs[None]),
        allocated_contacts=integer(info.max_contact_pairs[None]))
    require(integer(solver._options.max_collision_pairs)==row['max_collision_pairs_option'],
            'native option and effective solver capacity differ')
    return deepcopy(validate_contract(row))


class NativeContactGuard:
    def __init__(self,solver,runtime):
        self.solver=solver;self.runtime=runtime
        self.initial=None;self.terminal=None;self.attempted=0;self.passed=0
        self.failure=None;self.terminal_passed=False;self.finished=False
        try:
            self.initial=read_contract(solver,runtime);solver.check_errno()
        except Exception as error:
            self._failure('initial',error);raise

    def _failure(self,stage,error):
        if self.failure is None:
            self.failure=dict(stage=stage,before_recorded_sample=self.passed,error=repr(error))

    def before_sample(self,recorded_samples):
        require(not self.finished and self.failure is None,'native guard cannot resume after failure or termination')
        try:
            require(type(recorded_samples) is int and recorded_samples==self.passed==self.attempted
                and recorded_samples<MAX_PHYSICS_SAMPLES,'one native check for each ordered recorded sample')
            self.attempted+=1
            require(read_contract(self.solver,self.runtime)==self.initial,'native contact contract changed during acquisition')
            self.solver.check_errno()
            self.passed+=1
        except Exception as error:
            self._failure('before_sample',error);raise

    def finish(self,recorded_samples):
        require(not self.finished,'native terminal check cannot be repeated')
        self.finished=True
        try:
            self.terminal=read_contract(self.solver,self.runtime)
            self.solver.check_errno()  # Check even if the sample recorder failed.
            require(self.failure is None and self.terminal==self.initial,'native failure or terminal capacity drift')
            require(type(recorded_samples) is int and recorded_samples==self.passed==self.attempted,
                    'recorded prefix does not match native check coverage')
            self.terminal_passed=True
        except Exception as error:
            self._failure('terminal',error);raise

    def report(self):
        return dict(schema='independent_tracking_native_contact_integrity.v1',
            initial_contract=deepcopy(self.initial),terminal_contract=deepcopy(self.terminal),
            sample_checks_attempted=self.attempted,sample_checks_passed=self.passed,
            terminal_check_passed=self.terminal_passed,finished=self.finished,failure=deepcopy(self.failure),
            native_state_used_for_commands=False,navigation_qualified=False)


def from_scene(scene):
    import genesis as gs  # Only after the real constructor initialized Genesis.
    native_source_bindings()
    require(Path(gs.__file__).resolve()==NATIVE_ROOT/'__init__.py','loaded native package differs from reviewed source')
    return NativeContactGuard(scene.rigid_solver,gs)


def validate_report(report,physics_samples):
    require(type(report) is dict and set(report)=={
        'schema','initial_contract','terminal_contract','sample_checks_attempted','sample_checks_passed',
        'terminal_check_passed','finished','failure','native_state_used_for_commands','navigation_qualified'},
        'exact native contact-integrity report required')
    require(report['schema']=='independent_tracking_native_contact_integrity.v1'
        and report['failure'] is None and report['terminal_check_passed'] is True and report['finished'] is True
        and report['native_state_used_for_commands'] is False and report['navigation_qualified'] is False,
        'complete successful native error checks required, not array-size compliance alone')
    require(type(physics_samples) is int and 0<=physics_samples<=MAX_PHYSICS_SAMPLES
        and all(type(report[k]) is int and report[k]==physics_samples
                for k in ('sample_checks_attempted','sample_checks_passed')), 'native check coverage must match raw prefix')
    validate_contract(report['initial_contract']);validate_contract(report['terminal_contract'])
    require(report['initial_contract']==report['terminal_contract'],'native capacity changed at termination')
    return dict(native_error_check_coverage_verified=True,physics_samples=physics_samples,
                raw_contact_measurements_reconstructed=False,navigation_qualified=False)
