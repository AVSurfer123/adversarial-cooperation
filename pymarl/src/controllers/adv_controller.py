from modules.agents import REGISTRY as agent_REGISTRY, RNNAgent
from components.action_selectors import REGISTRY as action_REGISTRY
from controllers.basic_controller import BasicMAC
from utils.logging import get_logger
import torch
import os
import json

# This multi-agent controller shares parameters between agents
class AdvMAC(torch.nn.Module):
    def __init__(self, scheme, groups, args):
        super().__init__()
        self.logger = get_logger()
        # Load in fixed policy for N-1 agents
        self.args = args
        
        with open(os.path.join(args.trained_agent_policy, 'params.json'), 'r') as f:
            fixed_args = json.load(f)
        self.fixed_agents = BasicMAC(scheme, groups, fixed_args)
        self.fixed_agents.load_models(args.trained_agent_policy)
            
        # Create victim policy
        self.n_agents = 1
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep, :self.n_agents]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        other_actions = self.fixed_agents.select_actions(ep_batch, t_ep, t_env, bs, test_mode).detach()
        actions = torch.cat((chosen_actions, other_actions[:, 1:]), dim=1) # batch x n_agents
        return actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits, only for COMA
        if self.args.agent_output_type == "pi_logits":
            avail_actions = ep_batch["avail_actions"][:, t]

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = torch.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + torch.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        self.fixed_agents.init_hidden(batch_size)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.fixed_agents.cuda()

    def save_models(self, path):
        torch.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(torch.load("{}/agent.th".format(path)))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t, :self.n_agents])  # b1av
        if self.args.obs_last_action: # Add last action to observation
            if t == 0:
                inputs.append(torch.zeros_like(batch["actions_onehot"][:, t, :self.n_agents]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1, :self.n_agents])
        if self.args.obs_agent_id:  # Add agent number to observation (useless for adversarial case)
            inputs.append(torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = torch.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
