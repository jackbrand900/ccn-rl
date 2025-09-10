import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from src.models.network import ModularNetwork
from src.utils.preprocessing import prepare_input, prepare_batch
from src.utils.context_provider import build_context
from src.utils.constraint_monitor import ConstraintMonitor
from src.utils.shield_controller import ShieldController


class ConstrainedPPOAgent:
    def __init__(self, input_shape, action_dim,
                 agent_kwargs=None,
                 use_cnn=False,
                 env=None,
                 requirements_path=None,
                 monitor_constraints=True,
                 mode='hard',
                 budget=0.05,
                 nu_lr=1e-3,
                 verbose=False):

        self.env = env
        self.use_cnn = use_cnn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.memory = []
        self.monitor_constraints = monitor_constraints
        self.use_shield_pre = False
        self.use_shield_post = False
        self.use_shield_layer = False

        # === Hyperparameters ===
        agent_kwargs = agent_kwargs or {}
        agent_kwargs = {'lr': 0.0017, 'gamma': 0.97, 'hidden_dim': 512, 'use_orthogonal_init': True,
                        'num_layers': 4, 'cost_gamma': 0.96, 'cost_lam': 0.94, 'clip_eps': 0.18,
                        'ent_coef': 0.018, 'epochs': 1, 'batch_size': 32}
        self.hidden_dim = agent_kwargs.get("hidden_dim", 128)
        self.use_orthogonal_init = agent_kwargs.get("use_orthogonal_init", False)
        self.lr = agent_kwargs.get("lr", 3e-4)
        self.gamma = agent_kwargs.get("gamma", 0.99)
        self.cost_gamma = agent_kwargs.get("cost_gamma", 0.99)
        self.lam = agent_kwargs.get("lam", 0.95)
        self.cost_lam = agent_kwargs.get("cost_lam", 0.95)
        self.clip_eps = agent_kwargs.get("clip_eps", 0.2)
        self.ent_coef = agent_kwargs.get("ent_coef", 0.01)
        self.batch_size = agent_kwargs.get("batch_size", 64)
        self.epochs = agent_kwargs.get("epochs", 10)
        self.num_layers = agent_kwargs.get("num_layers", 3)
        self.lambda_sem = 0

        # === Networks ===
        self.policy_net = ModularNetwork(input_shape, action_dim, self.hidden_dim,
                                         self.num_layers, self.use_cnn, None,
                                         self.use_orthogonal_init).to(self.device)

        self.value_net = ModularNetwork(input_shape, 1, self.hidden_dim,
                                        self.num_layers, self.use_cnn, None,
                                        self.use_orthogonal_init).to(self.device)

        self.cost_value_net = ModularNetwork(input_shape, 1, self.hidden_dim,
                                             self.num_layers, self.use_cnn, None,
                                             self.use_orthogonal_init).to(self.device)

        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=self.lr)
        self.cost_optimizer = optim.Adam(self.cost_value_net.parameters(), lr=self.lr)

        # === Constraints ===
        self.constraint_monitor = ConstraintMonitor(verbose=verbose)
        self.shield_controller = ShieldController(requirements_path, action_dim, mode,
                                                  verbose=verbose, is_shield_active=False)
        self.shield_controller.constraint_monitor = self.constraint_monitor

        # === Lagrangian multiplier ===
        self.budget = budget
        self.nu = torch.tensor(1.0, requires_grad=True, device=self.device)
        self.nu_optim = optim.Adam([self.nu], lr=nu_lr)

    def compute_gae(self, rewards, values, dones, gamma, lam):
        advantages, gae = [], 0
        values = values + [0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return returns, advantages

    def select_action(self, state, env=None, do_apply_shield=True):
        self.last_obs = state
        context = build_context(env or self.env, self)

        state_tensor = prepare_input(state, use_cnn=self.use_cnn).to(self.device)
        # print("[DEBUG] state_tensor shape before CNN:", state_tensor.shape)

        logits = self.policy_net(state_tensor)
        if isinstance(logits, tuple):  # (logits, features)
            logits = logits[0]

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value_net(state_tensor)
        if isinstance(value, tuple):
            value = value[0]
        value = value.squeeze()
        raw_probs = torch.softmax(logits, dim=-1).squeeze(0)
        shielded_probs = self.shield_controller.apply(raw_probs.unsqueeze(0), context).squeeze(0)
        shielded_probs /= shielded_probs.sum()

        dist_shielded = Categorical(probs=shielded_probs)
        a_shielded = dist_shielded.sample().item()
        a_unshielded = action.item()

        self.constraint_monitor.log_step_from_probs_and_actions(
            raw_probs=raw_probs.detach(),
            corrected_probs=shielded_probs.detach(),
            a_unshielded=a_unshielded,
            a_shielded=a_shielded,
            context=context,
            shield_controller=self.shield_controller,
        )

        self.memory.append({
            'state': state,
            'action': action.item(),
            'reward': 0.0,
            'next_state': None,
            'done': False,
            'log_prob': log_prob.item(),
            'value': value.item(),
            'cost': 0.0,
            'context': context
        })

        return a_unshielded, a_unshielded, a_shielded, context

    def store_transition(self, state, action, reward, next_state, context, done):
        if not self.memory:
            raise ValueError("No memory to update. Call select_action first.")
        self.memory[-1]['reward'] = reward
        self.memory[-1]['next_state'] = next_state
        self.memory[-1]['done'] = done

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]

        states = prepare_batch([t['state'] for t in batch], use_cnn=self.use_cnn).to(self.device)
        actions = torch.tensor([t['action'] for t in batch]).to(self.device)
        rewards = [t['reward'] for t in batch]
        costs = [t['cost'] for t in batch]
        dones = [t['done'] for t in batch]
        old_log_probs = torch.tensor([t['log_prob'] for t in batch]).to(self.device)
        values = [t['value'] for t in batch]

        with torch.no_grad():
            cost_values = []
            for s in states:
                out = self.cost_value_net(s.unsqueeze(0))
                value = out[0] if isinstance(out, tuple) else out
                cost_values.append(value.item())

        returns, adv = self.compute_gae(rewards, values, dones, self.gamma, self.lam)
        cost_returns, cost_adv = self.compute_gae(costs, cost_values, dones, self.cost_gamma, self.cost_lam)

        adv = torch.tensor(adv).float().to(self.device)
        cost_adv = torch.tensor(cost_adv).float().to(self.device)
        returns = torch.tensor(returns).float().to(self.device)
        cost_returns = torch.tensor(cost_returns).float().to(self.device)

        # === PPO Policy update ===
        logits = self.policy_net(states)
        if isinstance(logits, tuple):  # (logits, features)
            logits = logits[0]

        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        cost_surr1 = ratio * cost_adv
        cost_surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * cost_adv
        cost_loss = torch.min(cost_surr1, cost_surr2).mean()

        total_loss = policy_loss + self.nu.clamp(min=0.0) * cost_loss - self.ent_coef * entropy
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        value_out = self.value_net(states)
        if isinstance(value_out, tuple):
            value_out = value_out[0]
        value_preds = value_out.squeeze()
        value_loss = nn.MSELoss()(value_preds, returns)
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

        cost_out = self.cost_value_net(states)
        if isinstance(cost_out, tuple):
            cost_out = cost_out[0]
        cost_value_preds = cost_out.squeeze()
        cost_value_loss = nn.MSELoss()(cost_value_preds, cost_returns)
        self.cost_optimizer.zero_grad()
        cost_value_loss.backward()
        self.cost_optimizer.step()

        avg_cost = torch.tensor(costs).float().mean().to(self.device)
        lagrangian_loss = -(self.nu * (avg_cost - self.budget).detach())
        self.nu_optim.zero_grad()
        lagrangian_loss.backward()
        self.nu_optim.step()

        self.memory.clear()

    def get_weights(self):
        return {
            "policy": self.policy_net.state_dict(),
            "value": self.value_net.state_dict(),
            "cost": self.cost_value_net.state_dict()
        }

    def load_weights(self, weights):
        self.policy_net.load_state_dict(weights["policy"])
        self.value_net.load_state_dict(weights["value"])
        self.cost_value_net.load_state_dict(weights["cost"])
