from modules.agents import REGISTRY as agent_REGISTRY, RNNAgent
from components.action_selectors import REGISTRY as action_REGISTRY
from controllers.basic_controller import BasicMAC
from utils.logging import get_logger
import math
import torch
from torch.autograd.gradcheck import zero_gradients

# This multi-agent controller shares parameters between agents
class AdvMAC:
    def __init__(self, scheme, groups, args):
        self.logger = get_logger()
        # Load in fixed policy for N-1 agents
        self.args = args
        self.fixed_agents = BasicMAC(scheme, groups, args)
        self.fixed_agents.load_models(args.trained_agent_policy)

        # Create victim policy
        self.agent_index = 0
        self.n_agents = 1
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep, :self.n_agents]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        # Adversarial action, batch x 1
        chosen_action = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)

        if self.args.adversarial == 'obs':
            # Use JSMA to modify observation to victim
            agent_inputs = self._build_inputs(ep_batch, t_ep)
            adv_input = self.adv_forward(agent_inputs, ep_batch.batch_size, chosen_action.squeeze()) # batch x 92

            agent_inputs = self.fixed_agents._build_inputs(ep_batch, t_ep)
            agent_inputs = agent_inputs.view(ep_batch.batch_size, self.fixed_agents.n_agents, -1) # batch x 5 x 96
            zeros = torch.zeros(self.fixed_agents.n_agents - self.n_agents).cuda().expand(ep_batch.batch_size, 1, -1) # batch x 1 x 4
            adv_input = torch.cat((adv_input.unsqueeze(1), zeros), dim=2) # batch x 1 x 96
            agent_inputs[:, self.agent_index] = adv_input
            agent_inputs = agent_inputs.view(-1, agent_inputs.shape[-1]) # batch*n_agents x 96

            actions = self.fixed_agents.select_actions(ep_batch, t_ep, t_env, bs, test_mode, agent_inputs=agent_inputs).detach()
        else:
            # Directly change victim's action
            actions = self.fixed_agents.select_actions(ep_batch, t_ep, t_env, bs, test_mode).detach()
            # actions[:, self.agent_index] = chosen_action[:, 0]
            actions = torch.cat((chosen_action, actions[:, 1:]), dim=1) # batch x n_agents
        return actions

    def forward(self, ep_batch, t, test_mode=False, agent_inputs=None):
        if agent_inputs is None:
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

    def adv_forward(self, agent_inputs, batch_size, target_class, agent_idx=0):
        agent_inputs = agent_inputs.view(batch_size, self.n_agents,-1)
        victim_input = self.jsma(agent_inputs[:, agent_idx], target_class=target_class, agent_idx=agent_idx)
        agent_inputs[:, agent_idx] = victim_input
        agent_inputs = agent_inputs.view(batch_size * self.n_agents, -1)
        return agent_inputs

    def compute_jacobian(self, inputs, output):
        """
        :param inputs: Batch X Size (e.g. Depth X Width X Height)
        :param output: Batch X Classes
        :return: jacobian: Batch X Classes X Size
        """
        assert inputs.requires_grad

        num_classes = output.size()[1]

        jacobian = torch.zeros(num_classes, *inputs.size())
        grad_output = torch.zeros(*output.size())
        if inputs.is_cuda:
            grad_output = grad_output.cuda()
            jacobian = jacobian.cuda()

        for i in range(num_classes):
            zero_gradients(inputs)
            grad_output.zero_()
            grad_output[:, i] = 1
            output.backward(grad_output, retain_graph=True)
            jacobian[i] = inputs.grad.data

        return torch.transpose(jacobian, dim0=0, dim1=1)


    def select_best_saliency_map(self, jacobian, target_index):
        # jacobian: actions x features
        num_features = jacobian.shape[-1]
        all_sum = torch.sum(jacobian, 0) # sum all q partial derivatives for each feature
        alpha = jacobian[target_index] # get target q partial derivative for each feature
        
        best_score = best_f1 = best_f2 = 0
        best_i = best_j = 0
        for i in range(num_features):
            for j in range(i+1, num_features):
                
                f1 = alpha[i] # dQ_t/dx_i
                f2 = alpha[j] # dQ_t/dx_j
                
                remaining_sum = all_sum[i]+all_sum[j]-f1-f2 # sum_k(dQ_k/dx_i) + sum_k(dQ_k/dx_j) - dQ_t/dx_i - dQ_t/dx_j
                saliency_score = (f1+f2)*remaining_sum # (dQ_t/dx_i+dQ_t/dx_j)*sum_(k != t)(dQ_k/dx_i+dQ_k/dx_j)

                if saliency_score > 0:
                    saliency_score = 0
                else:
                    saliency_score *= -1

                if saliency_score > best_score:
                    best_score = saliency_score
                    best_f1, best_f2 = f1, f2
                    best_i, best_j = i, j

        return best_i,best_j, math.copysign(1, best_f1+best_f2)
                

    def jsma(self, inputs, target_class, agent_idx=0):
        # inputs batch x features
        # Make a clone since we will alter the values
        input_features = torch.autograd.Variable(inputs.clone(), requires_grad=True)
        input_features.requires_grad = True

        hidden_states = self.hidden_states.view(-1, self.n_agents,self.hidden_states.shape[-1])[:, agent_idx]
        output, _ = self.agent(input_features, hidden_states)
        source_class = torch.argmax(output[0])
        theta = 0.1
        max_theta = 0.9
        k = 0
        max_iter = 20
        while source_class != target_class and k < max_iter:
            jacobian = self.compute_jacobian(input_features, output)
            i,j,d = self.select_best_saliency_map(jacobian.squeeze(0), target_class)

            perturbation = theta*d
            input_features.data[0,i] += perturbation
            input_features.data[0,j] += perturbation

            output, _ = self.agent(input_features, hidden_states)
            source_class = torch.argmax(output[0])
            theta = min(theta+.1,max_theta)
            k+=1
        if source_class != target_class:
            print("Changed source %d to target %d" % (source_class, target_class))
        return input_features
