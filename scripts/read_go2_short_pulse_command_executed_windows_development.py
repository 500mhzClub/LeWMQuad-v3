"""Same 36 recorded executions with the newly frozen pulse-trained control."""
from lewm.eligible_floor_registration_development import bind
from scripts import read_go2_command_history_executed_windows_development as reader
from scripts.fit_go2_short_pulse_command_control_development import OUTPUT as FIT

OUTPUT=reader.BASE/'go2_short_pulse_command_executed_windows_v1_attempt_001'


if __name__=='__main__':
    bind(reader.main,OUTPUT=OUTPUT,FIT=FIT)()
