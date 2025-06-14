import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import core.utils.utils as utils


def gaussian_logprob(noise, log_std):
    """
    Compute Gaussian log probability.
    """
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """
    Apply squashing function.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


class DeterministicActor(nn.Module):
    """
    Original TD3 actor.
    """
    def __init__(self, feature_dim, action_dim, hidden_dim, max_action):
        super(DeterministicActor, self).__init__()

        self.max_action = max_action

        self.policy1= nn.Linear(feature_dim, hidden_dim)
        self.policy2= nn.Linear(hidden_dim, hidden_dim)
        self.policy3= nn.Linear(hidden_dim, action_dim)

        self.apply(utils.weight_init)

    def forward(self, state):
        a = F.relu(self.policy1(state), inplace=True)
        a = F.relu(self.policy2(a), inplace=True)
        a = self.policy3(a)

        return torch.tanh(a)* self.max_action


class Critic(nn.Module):
    """
    Original TD3 critic.
    """
    def __init__(self, feature_dim, action_dim, hidden_dim):
        super().__init__()

        self.q1_layer1 = nn.Linear(feature_dim + action_dim, hidden_dim)
        self.q1_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_layer3 = nn.Linear(hidden_dim, 1)

        self.q2_layer1 = nn.Linear(feature_dim + action_dim, hidden_dim)
        self.q2_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_layer3 = nn.Linear(hidden_dim, 1)

        self.apply(utils.weight_init)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.q1_layer1(sa), inplace=True)
        q1 = F.relu(self.q1_layer2(q1), inplace=True)
        q1 = self.q1_layer3(q1)

        q2 = F.relu(self.q2_layer1(sa), inplace=True)
        q2 = F.relu(self.q2_layer2(q2), inplace=True)
        q2 = self.q2_layer3(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.q1_layer1(sa), inplace=True)
        q1 = F.relu(self.q1_layer2(q1), inplace=True)
        q1 = self.q1_layer3(q1)

        return q1
