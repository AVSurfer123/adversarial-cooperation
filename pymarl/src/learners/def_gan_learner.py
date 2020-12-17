import copy
from components.episode_buffer import EpisodeBatch
from modules.gan import DiscriminatorMLP, GeneratorMLP
import torch
from torch.nn import functional as F
from torch import nn, optim


class DefGANLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        if args.use_cuda:
            self.cuda()

        self.g_optimizer = optim.Adam(self.mac.G.parameters(), lr=args.lr, betas=(0, 0.9), eps=args.optim_eps)
        self.d_optimizer = optim.Adam(self.mac.D.parameters(), lr=args.lr, betas=(0, 0.9), eps=args.optim_eps)
        # self.g_scheduler = optim.lr_scheduler.LambdaLR(self.g_optimizer, lambda e: max(1 - e/30, .1))
        
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        torch.set_anomaly_enabled(True)
        # Get the relevant quantities
        # Need to limit number of agents for adversarial DQN
        rewards = batch["reward"][:, :-1] # batch x time x 1
        actions = batch["actions"][:, :-1, :self.mac.n_agents]
        terminated = batch["terminated"][:, :-1, :self.mac.n_agents].float()
        mask = batch["filled"][:, :-1, :self.mac.n_agents].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :, :self.mac.n_agents]

        eps = 1e-6

        for t in range(rewards.shape[1]):
            obs = batch['obs'][:, t, self.mac.agent_idx].detach() # batch x obs_size

            noise = self.mac.G.generate_noise(len(obs))
            generated = self.mac.G(noise)
            fake_discrim = self.mac.D(generated)
            real_discrim = self.mac.D(obs)
            d_loss = -(torch.log(real_discrim + eps) + torch.log(1 - fake_discrim + eps)).mean()  # Negate to change max to min
            self.d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            d_grad_norm = nn.utils.clip_grad_norm_(self.mac.D.parameters(), self.args.grad_norm_clip)
            self.d_optimizer.step()

            noise = self.mac.G.generate_noise(len(obs))
            fake_discrim = self.mac.D(self.mac.G(noise))
            g_loss = -torch.log(fake_discrim + eps).mean()
            self.g_optimizer.zero_grad()
            g_loss.backward()
            g_grad_norm = nn.utils.clip_grad_norm_(self.mac.G.parameters(), self.args.grad_norm_clip)
            self.g_optimizer.step()


        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat("gen_loss", g_loss.item(), t_env)
            self.logger.log_stat("discrim_loss", d_loss.item(), t_env)
            self.logger.log_stat("gen_grad_norm", g_grad_norm.item(), t_env)
            self.logger.log_stat("discrim_grad_norm", d_grad_norm.item(), t_env)
            # mask_elems = mask.sum().item()
            self.log_stats_t = t_env

    def cuda(self):
        self.mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        opts = {
            'G': self.g_optimizer.state_dict(),
            'D': self.d_optimizer.state_dict(),
        }
        torch.save(opts, "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        opts = torch.load("{}/opt.th".format(path))
        self.g_optimizer.load_state_dict(opts['G'])
        self.d_optimizer.load_state_dict(opts['D'])
