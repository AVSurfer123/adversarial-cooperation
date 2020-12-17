REGISTRY = {}

from .basic_controller import BasicMAC
from .adv_controller import AdvMAC
from .obs_controller import Obs_MAC
from .def_gan_controller import DEF_GAN

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["obs_mac"] = Obs_MAC
REGISTRY["def_gan_mac"] = DEF_GAN
