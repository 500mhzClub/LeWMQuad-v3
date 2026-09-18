"""Thirteen fresh acquisitions to locate the measured 220-ms camera bottleneck."""
import cProfile
import io
import pstats

from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_paced_native_prefix_development as source

OUTPUT=source.BASE/'go2_native_acquisition_profile_layout00_v1_attempt_001'


class ProfiledSession(source.PacedNativeSession):
    def sensor_packets(self):
        if not hasattr(self,'acquisition_profile'):self.acquisition_profile=cProfile.Profile()
        self.acquisition_profile.enable()
        try:return super().sensor_packets()
        finally:self.acquisition_profile.disable()

    def persist_observations(self,directory):
        super().persist_observations(directory)
        if hasattr(self,'acquisition_profile'):
            with (OUTPUT/'acquisition_profile.pstats').open('xb'):pass
            self.acquisition_profile.dump_stats(str(OUTPUT/'acquisition_profile.pstats'))
            stream=io.StringIO()
            pstats.Stats(self.acquisition_profile,stream=stream).sort_stats('cumulative').print_stats(35)
            with (OUTPUT/'acquisition_profile.txt').open('x') as f:f.write(stream.getvalue())


if __name__=='__main__':
    bind(source.main,OUTPUT=OUTPUT,COUNT=13,PacedNativeSession=ProfiledSession,
        write=bind(source.write,OUTPUT=OUTPUT))()
