REGISTRY = {}

from .basic_controller import BasicMAC
from .adv_controller import AdvMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["adv_mac"] = AdvMAC
