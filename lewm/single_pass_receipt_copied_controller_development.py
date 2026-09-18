"""Compose existing exact index and receipt-copy candidates at empty state."""
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportMemory, MeasuredFloorTransportMap
from lewm.receipt_copied_selector_development import ReceiptCopiedMeasuredFloorController
from lewm.single_pass_sample_bounds_development import SinglePassMeasuredSampleBoundsIndex


class SinglePassMeasuredFloorMemory(MeasuredFloorTransportMemory):
    def __init__(self, *, identity):
        super().__init__(identity=identity)
        self.index = SinglePassMeasuredSampleBoundsIndex()
        self.auxiliary_index = SinglePassMeasuredSampleBoundsIndex()
        for partition in (self.partition, self.auxiliary_partition, self.confirmed_auxiliary_partition):
            partition.floor = SinglePassMeasuredSampleBoundsIndex()
            partition.other = SinglePassMeasuredSampleBoundsIndex()


class SinglePassMeasuredFloorMap(MeasuredFloorTransportMap):
    def __init__(self, *, identity=(0,0,0)):
        super().__init__(identity=identity)
        self.surface = SinglePassMeasuredFloorMemory(identity=identity)


class SinglePassReceiptCopiedController(ReceiptCopiedMeasuredFloorController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapper = SinglePassMeasuredFloorMap(identity=(0,0,0))
        self.memory = self.mapper.surface
