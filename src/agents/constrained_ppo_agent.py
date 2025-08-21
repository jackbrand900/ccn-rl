# Constrained PPO (CMDP-style) Agent with Shielding

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from src.utils.preprocessing import prepare_input
import src.utils.context_provider as context_provider
from src.utils.shield_controller import ShieldController
from src.utils.constraint_monitor import ConstraintMonitor


class ConstrainedPPOAgent:
    def __init__(self, policy, value_net, cost_value_net,
                 input_shape, action_dim,
                 lr=3e-4, gamma=0.99, cost_gamma=0.99, lam=0.95, cost_lam=0.95,
                 clip_eps=0.2, ent_coef=0.01, budget=0.05, nu_lr=1e-3,
                 use_cnn=False, use_shield_post=False, use_shield_pre=False, use_shield_layer=False,
                 monitor_constraints=True, requirements_path=None, env=None, mode='hard', verbose=False):

        self.policy = policy
        self.value_net = value_net
        self.cost_value_net = cost_value_net

        self.optimizer = optim.Adam(list(policy.parameters()) + list(value_net.parameters()), lr=lr)
        self.cost_optimizer = optim.Adam(cost_value_net.parameters(), lr=lr)

        self.gamma = gamma
        self.cost_gamma = cost_gamma
        self.lam = lam
        self.cost_lam = cost_lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef

        self.budget = budget
        self.nu = torch.tensor(1.0, requires_grad=True)
        self.nu_optim = optim.Adam([self.nu], lr=nu_lr)

        self.env = env
        self.use_cnn = use_cnn
        self.action_dim = action_dim
        self.memory = []

        self.use_shield_post = use_shield_post
        self.use_shield_pre = use_shield_pre
        self.use_shield_layer = use_shield_layer
        self.monitor_constraints = monitor_constraints
        self.constraint_monitor = ConstraintMonitor(verbose=verbose)
        self.shield_controller = ShieldController(
            requirements_path, action_dim, mode, verbose=verbose,
            is_shield_active=use_shield_layer or use_shield_post or use_shield_pre
        )
        self.shield_controller.constraint_monitor = self.constraint_monitor

    def compute_gae(self, rewards, values, dones, gamma, lam):
        advantages = []
        gae = 0
        values = values + [0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return returns, advantages

    def select_action(self, state, env=None, do_apply_shield=True):
        context = context_provider.build_context(env or self.env, self)
        state_tensor = prepare_input(state, use_cnn=self.use_cnn).to(next(self.policy.parameters()).device)

        action, log_prob, value, raw_probs, shielded_probs = self.policy.select_action(
            state_tensor, context,
            constraint_monitor=self.constraint_monitor if self.use_shield_layer else None
        )

        if self.use_shield_layer:
            a_shielded = action
            dist_unshielded = Categorical(probs=raw_probs)
            a_unshielded = dist_unshielded.sample().item()
            selected_action = a_shielded
            log_prob_tensor = log_prob

        elif self.use_shield_post or self.use_shield_pre and do_apply_shield:
            dist_unshielded = Categorical(probs=raw_probs)
            a_unshielded = dist_unshielded.sample().item()

            shielded_probs = self.shield_controller.apply(raw_probs.unsqueeze(0), context).squeeze(0)
            shielded_probs /= shielded_probs.sum()
            dist_shielded = Categorical(probs=shielded_probs)
            a_shielded = dist_shielded.sample().item()

            selected_action = a_shielded
            log_prob_tensor = dist_shielded.log_prob(torch.tensor(a_shielded).to(raw_probs.device))

        else:
            dist_unshielded = Categorical(probs=raw_probs)
            a_unshielded = dist_unshielded.sample().item()
            log_prob_tensor = dist_unshielded.log_prob(torch.tensor(a_unshielded).to(raw_probs.device))

            if self.monitor_constraints:
                shielded_probs = self.shield_controller.apply(raw_probs.unsqueeze(0), context).squeeze(0)
                shielded_probs /= shielded_probs.sum()
                dist_shielded = Categorical(probs=shielded_probs)
                a_shielded = dist_shielded.sample().item()
            else:
                a_shielded = None

            selected_action = a_unshielded

        violation_cost = self.constraint_monitor.compute_instant_cost(
            context=context,
            a_unshielded=a_unshielded,
            a_applied=selected_action,
        ) if self.monitor_constraints else 0.0

        self.memory.append({
            'state': state,
            'action': selected_action,
            'reward': 0.0,
            'next_state': None,
            'done': False,
            'log_prob': log_prob_tensor.item(),
            'value': value.item(),
            'cost': violation_cost,
            'context': context
        })

        return selected_action, a_unshielded, a_shielded, context, violation_cost

    def update(self):
        if len(self.memory) == 0:
            return

        states = torch.stack([prepare_input(t['state'], use_cnn=self.use_cnn) for t in self.memory])
        actions = torch.tensor([t['action'] for t in self.memory])
        rewards = [t['reward'] for t in self.memory]
        costs = [t['cost'] for t in self.memory]
        dones = [t['done'] for t in self.memory]
        old_log_probs = torch.tensor([t['log_prob'] for t in self.memory])

        values = [t['value'] for t in self.memory]
        with torch.no_grad():
            cost_values = [self.cost_value_net(s.unsqueeze(0)).item() for s in states]

        returns, adv = self.compute_gae(rewards, values, dones, self.gamma, self.lam)
        cost_returns, cost_adv = self.compute_gae(costs, cost_values, dones, self.cost_gamma, self.cost_lam)

        adv = torch.tensor(adv, dtype=torch.float32)
        cost_adv = torch.tensor(cost_adv, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        cost_returns = torch.tensor(cost_returns, dtype=torch.float32)

        dist = self.policy(states)
        new_log_probs = dist.log_prob(actions)
        ratio = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        cost_surr1 = ratio * cost_adv
        cost_surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * cost_adv
        cost_loss = torch.min(cost_surr1, cost_surr2).mean()

        entropy = dist.entropy().mean()
        total_loss = policy_loss + self.nu.clamp(min=0.0) * cost_loss - self.ent_coef * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        value_preds = self.value_net(states).squeeze()
        value_loss = nn.MSELoss()(value_preds, returns)
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

        cost_value_preds = self.cost_value_net(states).squeeze()
        cost_value_loss = nn.MSELoss()(cost_value_preds, cost_returns)
        self.cost_optimizer.zero_grad()
        cost_value_loss.backward()
        self.cost_optimizer.step()

        avg_cost = torch.mean(torch.tensor(costs))
        lagrangian_loss = -(self.nu * (avg_cost - self.budget).detach())
        self.nu_optim.zero_grad()
        lagrangian_loss.backward()
        self.nu_optim.step()

        self.memory.clear()
