REGISTRY = {}

from .basic_controller import BasicMAC
from .adv_controller import AdvMAC
from .gan_controller import GAN_MAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["adv_mac"] = AdvMAC
REGISTRY["gan_mac"] = GAN_MAC
