from modules.gan import DiscriminatorMLP, GeneratorMLP
from components.action_selectors import REGISTRY as action_REGISTRY
from controllers import BasicMAC, AdvMAC
from utils.logging import get_logger
import torch
from torch import nn, optim
import os
import json


# This multi-agent controller shares parameters between agents
class Obs_MAC(nn.Module):
    def __init__(self, scheme, groups, args):
        super().__init__()
        self.logger = get_logger()
        # Load in fixed policy for N-1 agents
        self.args = args
        self.scheme = scheme

        try:
            with open(os.path.join(args.trained_agent_policy, 'params.json'), 'r') as f:
                fixed_args = json.load(f)
        except:
            fixed_args = self.args
        self.fixed_agents = BasicMAC(scheme, groups, fixed_args)
        self.fixed_agents.load_models(args.trained_agent_policy)

        try:
            with open(os.path.join(args.trained_adv_policy, 'params.json'), 'r') as f:
                adv_args = json.load(f)
        except:
            adv_args = self.args
        self.adv_policy = AdvMAC(scheme, groups, adv_args)
        self.adv_policy.load_models(args.trained_adv_policy)

        # Create victim policy
        self.n_agents = 1
        self.agent_idx = 0

        obs_size = scheme['obs']['vshape']
        self.G = GeneratorMLP(obs_size, args.gan_hidden_size, obs_size)
        # self.D = DiscriminatorMLP(args.gan_hidden_size, obs_size)
        self.action_selector = action_REGISTRY[args.action_selector](args)


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][bs, t_ep]
        # random_actions = torch.distributions.Categorical(avail_actions.float()).sample().long()
        adv_action, _, qvals, chosen_action = self.forward(ep_batch, t_ep, test_mode=test_mode)
        acc = (adv_action == chosen_action).sum()
        print("Controlled {}/{} actions with obs perturbation".format(acc, len(adv_action)))
        actions = self.action_selector.select_action(qvals[bs], avail_actions, t_env, test_mode=test_mode)
        return actions

    def forward(self, ep_batch, t, test_mode=False):
        adv_qvals = self.adv_policy(ep_batch, t, test_mode) # batch x 1 x num_actions
        adv_action = torch.argmax(adv_qvals.squeeze(1), dim=1, keepdim=True) # batch x 1
        obs = ep_batch['obs'][:, t, self.agent_idx] # batch x obs_size
        conditioned_obs = torch.cat((obs, adv_action.float()), dim=1)
        generated = self.G(conditioned_obs)

        ep_batch['obs'][:, t, self.agent_idx] = generated
        action_qvals = self.fixed_agents.forward(ep_batch, t, test_mode) # batch x n_agents x num_actions
        chosen_action = torch.argmax(action_qvals, dim=2, keepdim=True)[:, self.agent_idx] # batch x 1

        return adv_action, generated, action_qvals, chosen_action

    # Needed since runner calls this
    def init_hidden(self, batch_size):
        self.fixed_agents.init_hidden(batch_size)
        self.adv_policy.init_hidden(batch_size)

    def parameters(self):
        return list(self.G.parameters())

    def cuda(self):
        self.G.cuda()
        self.fixed_agents.cuda()
        self.adv_policy.cuda()

    def save_models(self, path):
        state = {
            'G': self.G.state_dict(),
        }
        torch.save(state, "{}/gan.th".format(path))

    def load_models(self, path):
        state = torch.load("{}/gan.th".format(path))
        self.G.load_state_dict(state['G'])
