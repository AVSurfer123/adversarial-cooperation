from copy import deepcopy
from numpy import float32
import os
from gym.spaces import Tuple, Box, Discrete

from pettingzoo.butterfly import pistonball_v0, cooperative_pong_v1
from supersuit import normalize_obs_v0, dtype_v0, color_reduction_v0

import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env import PettingZooEnv
from ray.tune import grid_search
from ray.tune.registry import register_env
from ray.rllib.env.group_agents_wrapper import _GroupAgentsWrapper
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.agents.qmix.qmix_policy import ENV_STATE



if __name__ == "__main__":
    """For this script, you need:
    1. Algorithm name and according module, e.g.: "PPo" + agents.ppo as agent
    2. Name of the aec game you want to train on, e.g.: "pistonball".
    3. num_cpus
    4. num_rollouts
    Does require SuperSuit
    """
    alg_name = "QMIX"

    # function that outputs the environment you wish to register.
    def env_creator(config):
        # env = pistonball_v0.env(continuous=False,local_ratio=config.get("local_ratio", 0.2))
        env = cooperative_pong_v1.env()
        
        agent_list = env.agents
        grouping = {
            "group_1": agent_list,
        }

        env = dtype_v0(env, dtype=float32)
        env = color_reduction_v0(env, mode="R")
        env = normalize_obs_v0(env)

        env = PettingZooEnv(env)
        
        obs_space = Tuple([env.observation_spaces[i] for i in agent_list])
        act_space = Tuple([env.action_spaces[i] for i in agent_list])
        print(env.observation_spaces)
        print(env.action_spaces)

        env = _GroupAgentsWrapper(env, grouping, obs_space=obs_space, act_space=act_space)
        return env


    num_cpus = 1
    num_rollouts = 2
    env_name = "cooperative_pong"

    # 1. Gets default training configuration and specifies the POMgame to load.
    config = deepcopy(get_agent_class(alg_name)._default_config)

    # 2. Set environment config. This will be passed to
    # the env_creator function via the register env lambda below.
    # config["env_config"] = {"local_ratio": 0.5}

    #obs_space = Tuple([Box(200,120) for i in range(20)])

    # 3. Register env
    register_env(env_name, lambda config: env_creator(config))
    
    test_env = env_creator({})
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    print(obs_space)
    print(act_space)

    # 6. Initialize ray and trainer object
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    config["log_level"] = "DEBUG"
    config["num_workers"] = 1
    # Fragment length, collected at once from each worker and for each agent!
    config["rollout_fragment_length"] = 30
    # Training batch size -> Fragments are concatenated up to this point.
    config["train_batch_size"] = 200
    # After n steps, force reset simulation
    config["horizon"] = 200
    # Default: False
    config["no_done_at_end"] = False
    # Info: If False, each agents trajectory is expected to have
    # maximum one done=True in the last step of the trajectory.
    # If no_done_at_end = True, environment is not resetted
    # when dones[__all__]= True.

    config["env"] = env_name
    config["framework"] = "torch"


    # 6. Initialize ray and trainer object
    ray.init(num_cpus=num_cpus + 1)
    # trainer = get_agent_class(alg_name)(env="pistonball", config=config)

    stop = {
        # "episode_reward_mean": args.stop_reward,
        "timesteps_total": 50000,
    }

    results = tune.run(alg_name, stop=stop, config=config, verbose=1, checkpoint_at_end=True)
    ray.shutdown()
