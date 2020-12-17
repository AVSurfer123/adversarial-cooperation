import copy
from components.episode_buffer import EpisodeBatch
from modules.gan import DiscriminatorMLP, GeneratorMLP
import torch
from torch.nn import functional as F
from torch import nn, optim


class ObsLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = self.mac.parameters()
        self.optimizer = optim.Adam(self.params, lr=args.lr)

        # self.g_optimizer = optim.Adam(self.mac.G.parameters(), lr=8e-4, betas=(0, 0.9), eps=args.optim_eps)
        # self.d_optimizer = optim.Adam(self.mac.D.parameters(), lr=8e-4, betas=(0, 0.9), eps=args.optim_eps)
        # self.g_scheduler = optim.lr_scheduler.LambdaLR(self.g_optimizer, lambda e: max(1 - e/30, .1))
        
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        # Need to limit number of agents for adversarial DQN
        rewards = batch["reward"][:, :-1] # batch x time x 1
        actions = batch["actions"][:, :-1, :self.mac.n_agents]
        terminated = batch["terminated"][:, :-1, :self.mac.n_agents].float()
        mask = batch["filled"][:, :-1, :self.mac.n_agents].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :, :self.mac.n_agents]

        total_loss = torch.zeros(1).cuda()
        self.mac.init_hidden(batch.batch_size)
        agent_idx = 0

        for t in range(rewards.shape[1]):
            adv_action, generated, action_qvals, chosen_action = self.mac(batch, t)

            adv_qvals = torch.gather(action_qvals[:, agent_idx], 1, adv_action).squeeze(1)
            chosen_qvals = torch.gather(action_qvals[:, agent_idx], 1, chosen_action).squeeze(1)

            loss = F.margin_ranking_loss(adv_qvals, chosen_qvals, torch.ones(len(adv_qvals)).cuda())
            total_loss += loss

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimizer.step()

        # g_loss = -torch.log(fake_discrim).mean()
        # d_loss = -(torch.log(real_discrim) + torch.log(1 - fake_discrim)).mean()  # Negate to change max to min
        # self.d_optimizer.zero_grad()
        # d_loss.backward(retain_graph=True)
        # self.d_optimizer.step()
        # self.g_optimizer.zero_grad()
        # g_loss.backward()
        # self.g_optimizer.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            # mask_elems = mask.sum().item()
            # self.logger.log_stat("adv_q_mean", (adv_qvals * mask).sum().item()/(mask_elems * self.mac.n_agents), t_env)
            # self.logger.log_stat("chosen_q_mean", (chosen_qvals * mask).sum().item()/(mask_elems * self.mac.n_agents), t_env)
            self.log_stats_t = t_env

    def cuda(self):
        self.mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        torch.save(self.optimizer.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.optimizer.load_state_dict(torch.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
