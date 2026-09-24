"""Same native robot-visible acquisition with a required auxiliary public packet."""
from scripts.auxiliary_depth_visible_robot_session_development import VisibleRobotFamilySession
from scripts.auxiliary_downward45_depth_capture_development import capture
from scripts.auxiliary_downward45_packet_replay_development import packet,public_acquisition
from scripts.run_go2_successive_choice_maze_development_v1 import write_json


class AuxiliaryDownward45GoalSession(VisibleRobotFamilySession):
    def __init__(self,*args,**kwargs):
        self.auxiliary_audit=[]
        super().__init__(*args,**kwargs)

    def sensor_packets(self):
        policy,depth,fast,now=super().sensor_packets()
        index=len(self.model_manifest)-1
        if not 0<=index<=253:raise ValueError('bounded native auxiliary goal observation required')
        if len(self.auxiliary_audit)==index:
            row=capture(self,self.output,index)
            if row['physical_sample_index']!=749+50*index or row['measured_ns']!=now:
                raise ValueError('exact primary/auxiliary physical sample pairing required')
            self.auxiliary_audit.append(row)
        if len(self.auxiliary_audit)!=index+1:raise ValueError('uninterrupted auxiliary acquisition required')
        auxiliary=packet(self.output,index,policy,public_acquisition(self.auxiliary_audit[index]),now_ns=now)
        return policy,depth,fast,auxiliary,now

    def persist_observations(self,output):
        super().persist_observations(output)
        write_json(output/'auxiliary_camera_audit.json',self.auxiliary_audit)

