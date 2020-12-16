REGISTRY = {}

from .basic_controller import BasicMAC
from .adv_controller import AdvMAC
from .jsma_controller import JsmaMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["adv_mac"] = AdvMAC
REGISTRY["jsma_mac"] = JsmaMAC
