"""
Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
https://arxiv.org/abs/1802.09477
"""

import copy
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F

from agents.core import DeterministicActor, Critic
from agents.agent_utils import * 


class TD3Agent:
    def __init__(self, obs_shape, act_shape, act_limit_low, act_limit_high, device, lr, hidden_dim,
                 critic_target_tau, num_expl_steps, update_every_steps,
                 stddev_schedule, stddev_clip, discount):

        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.obs_dim= obs_shape
        self.act_dim = act_shape
        self.act_limit_low= act_limit_low
        self.act_limit_high= act_limit_high
        self.hidden_dim = hidden_dim
        self.lr = lr
        #TODO: Check with Sahand
        self.discount= discount

        # models
        self.actor = DeterministicActor(self.obs_dim, self.act_dim, hidden_dim, self.act_limit_high).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(self.obs_dim, self.act_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.actor_target.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        with torch.no_grad():
            obs = torch.as_tensor(obs, device=self.device)
            stddev = schedule(self.stddev_schedule, step)

            if not eval_mode:
                action = self.actor(obs.float().unsqueeze(0)).detach().cpu().numpy()[0]
                action = action + np.random.normal(0, 0.5* self.act_limit_high*stddev, size=self.act_dim)
                if step < self.num_expl_steps:
                    action = np.random.uniform(self.act_limit_low, self.act_limit_high, size=self.act_dim)
            else:
                action= self.actor(obs.float().unsqueeze(0)).detach().cpu().numpy()[0]
            return action.astype(np.float32).clip(self.act_limit_low, self.act_limit_high)

    def observe(self, obs, action):
        obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        action = torch.as_tensor(action, device=self.device).float().unsqueeze(0)

        q, _ = self.critic(obs, action)

        return {
            'state': obs.cpu().numpy()[0],
            'value': q.detach().cpu().numpy()[0]
        }

    def update_critic(self, obs, action, reward, next_obs, mask, step):
        metrics = dict()

        reward= reward.unsqueeze(1)
        mask= mask.unsqueeze(1)


        with torch.no_grad():
            # Select action according to policy and add clipped noise
            stddev = schedule(self.stddev_schedule, step)
            noise = (torch.randn_like(action) * stddev).clamp(-self.stddev_clip, self.stddev_clip)

            next_action = (self.actor_target(next_obs) + noise).clamp(self.act_limit_low, self.act_limit_high)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + mask* self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = current_Q1.mean().item()
        metrics['critic_q2'] = current_Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        # Optimize the critic
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        # Compute actor loss
        actor_loss = -self.critic.Q1(obs, self.actor(obs)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        metrics['actor_loss'] = actor_loss.item()

        return metrics

    def update(self, batch, step):
        metrics = dict()

        batch= to_float_tensor(batch)
        obs, action, reward, next_obs, mask = to_torch(
            batch, self.device)

        obs = obs.float()
        next_obs = next_obs.float()

        metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(self.update_critic(obs, action, reward, next_obs, mask, step))

        # update actor (delayed)
        if step % self.update_every_steps == 0:
            metrics.update(self.update_actor(obs.detach(), step))

            # update target networks
            soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
            soft_update_params(self.actor, self.actor_target, self.critic_target_tau)

        return metrics

    def save(self, model_dir, step):
        model_save_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')
        model_save_dir.mkdir(exist_ok=True, parents=True)

        torch.save(self.actor.state_dict(), f'{model_save_dir}/actor.pt')
        torch.save(self.critic.state_dict(), f'{model_save_dir}/critic.pt')

    def load(self, model_dir, step):
        print(f"Loading the model from {model_dir}, step: {step}")
        model_load_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')

        self.actor.load_state_dict(
            torch.load(f'{model_load_dir}/actor.pt', map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(f'{model_load_dir}/critic.pt', map_location=self.device)
        )
