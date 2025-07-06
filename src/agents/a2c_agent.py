import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

from src.utils.shield_controller import ShieldController
import src.utils.context_provider as context_provider

class A2CNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy_head(x), self.value_head(x)


class A2CAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3, gamma=0.99,
                 use_shield=True, verbose=False, requirements_path=None, env=None):

        self.gamma = gamma
        self.use_shield = use_shield
        self.verbose = verbose
        self.env = env
        self.learn_step_counter = 0

        self.model = A2CNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.memory = []
        self.training_logs = {"policy_loss": [], "value_loss": [], "entropy": [], "prob_shift": []}

        if use_shield:
            self.shield_controller = ShieldController(
                requirements_path=requirements_path,
                num_actions=action_dim,
                flag_logic_fn=context_provider.key_flag_logic,
            )
        else:
            self.shield_controller = None

    def select_action(self, state, env=None, do_apply_shield=True):
        context = context_provider.build_context(env or self.env, self)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits, value = self.model(state_tensor)
        raw_probs = torch.softmax(logits, dim=-1)

        if do_apply_shield and self.shield_controller:
            shielded_probs = self.shield_controller.apply(raw_probs, context, self.verbose)
        else:
            shielded_probs = raw_probs

        # Detect if the shield changed the probabilities
        was_modified = not torch.allclose(raw_probs, shielded_probs, atol=1e-6)

        dist = Categorical(probs=shielded_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.last_log_prob = log_prob
        self.last_value = value.squeeze(0)
        self.last_raw_probs = raw_probs.squeeze(0).detach()
        self.last_shielded_probs = shielded_probs.squeeze(0).detach()

        return action.item(), context, was_modified

    def store_transition(self, state, action, reward, next_state, context, done):
        self.memory.append((state, action, reward, next_state, context, done,
                            self.last_log_prob, self.last_value,
                            self.last_raw_probs, self.last_shielded_probs))

    def update(self):
        if not self.memory:
            return

        self.learn_step_counter += 1

        states, actions, rewards, next_states, contexts, dones, log_probs, values, raw_probs, shielded_probs = zip(*self.memory)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        values = torch.stack(values)
        raw_probs = torch.stack(raw_probs)
        shielded_probs = torch.stack(shielded_probs)

        with torch.no_grad():
            _, next_values = self.model(torch.FloatTensor(np.array(next_states)))
            targets = rewards + self.gamma * next_values.squeeze() * (1 - dones)
            advantages = targets - values.squeeze()

        logits, predicted_values = self.model(states)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        policy_loss = -(advantages.detach() * new_log_probs).mean()
        value_loss = nn.MSELoss()(predicted_values.squeeze(), targets)

        # Constraint losses (optional for PiShield alignment)
        goal = torch.zeros_like(shielded_probs)
        goal.scatter_(1, actions.unsqueeze(1), 1.0)
        req_loss = nn.BCELoss()(shielded_probs, goal)
        consistency_loss = nn.MSELoss()(torch.softmax(logits, dim=-1), shielded_probs)

        loss = policy_loss + value_loss + 0.05 * req_loss + 0.05 * consistency_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_logs["policy_loss"].append(policy_loss.item())
        self.training_logs["value_loss"].append(value_loss.item())
        self.training_logs["entropy"].append(entropy.item())
        self.training_logs["prob_shift"].append((shielded_probs - raw_probs).abs().mean().item())

        self.memory.clear()
