"""New inventory initializer, unchanged tested native recorder/guard chain."""
from lewm.independent_layout_collection_development import CollectionInventory
from scripts.independent_layout_physical_init_development import InventoryPhysicalInit
from scripts.independent_pulse_context_session_development import PulseContextSession
from scripts.near_field_rgbd_capture_development import NearFieldCapture


class InventorySession(PulseContextSession,InventoryPhysicalInit):
    capture_fixed_rgb=NearFieldCapture.capture_fixed_rgb

    def __init__(self,inventory,spec,output):
        if not isinstance(inventory,CollectionInventory):raise ValueError('validated frozen construction inventory required')
        self.inventory=inventory
        super().__init__(spec,output)
