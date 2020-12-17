from modules.gan import DiscriminatorMLP, GeneratorMLP
from components.action_selectors import REGISTRY as action_REGISTRY
from controllers import BasicMAC, AdvMAC
from utils.logging import get_logger
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import time


# This multi-agent controller shares parameters between agents
class DEF_GAN(nn.Module):
    def __init__(self, scheme, groups, args):
        super().__init__()
        self.logger = get_logger()
        # Load in fixed policy for N-1 agents
        self.args = args
        self.scheme = scheme

        self.fixed_agents = BasicMAC(scheme, groups, args)
        self.fixed_agents.load_models(args.trained_agent_policy)

        # Create victim policy
        self.n_agents = 1
        self.agent_idx = 0

        obs_size = scheme['obs']['vshape']
        # -1 hack due to model re-use from obs_learner
        self.G = GeneratorMLP(args.gan_noise_size - 1, args.gan_hidden_size, obs_size) 
        self.D = DiscriminatorMLP(args.gan_hidden_size, obs_size - 1)
        self.action_selector = action_REGISTRY[args.action_selector](args)


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Remove noise when testing, else try to train from base data distribution
        if test_mode:
            obs = ep_batch['obs'][:, t_ep, self.agent_idx] # batch x obs_size
            z_orig, z_desc = self.get_z_sets(self.G, obs)
            z_star = self.get_z_star(self.G, obs, z_desc)
            generated = self.G(z_star)
            ep_batch['obs'][:, t_ep, self.agent_idx] = generated

        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.fixed_agents.forward(ep_batch, t_ep, test_mode=test_mode) # Use QMIX agent to get true data distribution
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        return

    # Needed since runner calls this
    def init_hidden(self, batch_size):
        self.fixed_agents.init_hidden(batch_size)

    def parameters(self):
        return list(self.G.parameters()) + list(self.D.parameters())

    def cuda(self):
        self.G.cuda()
        self.D.cuda()
        self.fixed_agents.cuda()

    def save_models(self, path):
        state = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
        }
        torch.save(state, "{}/gan.th".format(path))

    def load_models(self, path):
        state = torch.load("{}/gan.th".format(path))
        self.G.load_state_dict(state['G'])
        self.D.load_state_dict(state['D'])



    def adjust_lr(self, optimizer, cur_lr, decay_rate = 0.1, global_step = 1, rec_iter = 200):
        lr = cur_lr * decay_rate ** (global_step / int(np.ceil(rec_iter * 0.8)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr

    def get_z_sets(self, model, data, rec_iter=100, rec_rr=10, global_step=1):
        """

        To get R random different initializations of z from L steps of Gradient Descent.
        rec_iter : the number of L of Gradient Descent steps 
        rec_rr : the number of different random initialization of z

        """        
        # the output of R random different initializations of z from L steps of GD
        z_hats_recs = torch.Tensor(rec_rr, data.size(0), self.G.noise_size)
        
        # the R random differernt initializations of z before L steps of GD
        z_hats_orig = torch.Tensor(rec_rr, data.size(0), self.G.noise_size)
        
        lr = 1.0
        data = data.detach()

        for idx in range(rec_rr):
            
            z_hat = torch.randn(data.size(0), self.G.noise_size, device='cuda')
            z_hat = z_hat.detach().requires_grad_()
            
            cur_lr = lr
            optimizer = optim.SGD([z_hat], lr=cur_lr, momentum=0.7)
            z_hats_orig[idx] = z_hat.cpu().detach().clone()
            start_time = time.time()
            for iteration in range(rec_iter):
                optimizer.zero_grad()
                fake = model(z_hat)
                reconstruct_loss = F.mse_loss(fake, data)
                reconstruct_loss.backward()
                optimizer.step()
                cur_lr = self.adjust_lr(optimizer, cur_lr, global_step=global_step, rec_iter=rec_iter)
            # print("End time for iteration %d:" % idx, time.time() - start_time)
            z_hats_recs[idx] = z_hat.cpu().detach().clone()

        return z_hats_orig, z_hats_recs

    def get_z_star(self, model, data, z_hats_recs):
        """

        To get z* so as to minimize reconstruction error between generator G and an image x

        """
        
        error = torch.Tensor(len(z_hats_recs), data.shape[0])
        for i in range(len(z_hats_recs)):
            gen = model(z_hats_recs[i].cuda()) # batch x obs_size
            error[i] = ((gen - data)**2).sum(dim=1) # MSE loss
        min_idx = torch.argmin(error, dim=0) # batch
        min_idx = min_idx.unsqueeze(0).unsqueeze(-1) # 1 x batch x 1
        min_idx = min_idx.repeat(1, 1, z_hats_recs.shape[-1]) # 1 x batch x hidden_size
        z_star = torch.gather(z_hats_recs, 0, min_idx) # 1 x batch x hidden_size
        return z_star.squeeze(0)
