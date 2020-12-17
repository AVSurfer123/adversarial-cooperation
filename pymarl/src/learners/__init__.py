from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .obs_learner import ObsLearner
from .def_gan_learner import DefGANLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["obs_learner"] = ObsLearner
REGISTRY["def_gan_learner"] = DefGANLearner
