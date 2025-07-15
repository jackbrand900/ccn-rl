import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

from src.utils.shield_controller import ShieldController
import src.utils.context_provider as context_provider


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value


class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=3e-4, gamma=0.99, clip_eps=0.2,
                 ent_coef=0.01, lambda_req=0.05, lambda_consistency=0.05, use_shield=True,
                 verbose=False, requirements_path=None, env=None):

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.lambda_req = lambda_req
        self.lambda_consistency = lambda_consistency
        self.use_shield = use_shield
        self.verbose = verbose
        self.env = env
        self.action_dim = action_dim

        self.policy = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = []

        self.learn_step_counter = 0  # needed for context_provider
        self.last_log_prob = None
        self.last_value = None
        self.last_raw_probs = None
        self.last_shielded_probs = None
        self.last_obs = None

        if use_shield:
            self.shield_controller = ShieldController(
                requirements_path=requirements_path,
                num_actions=action_dim,
                flag_logic_fn=context_provider.cartpole_emergency_flag_logic,
            )
        else:
            self.shield_controller = None

        self.training_logs = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "req_loss": [],
            "consistency_loss": [],
            "prob_shift": [],
        }

    def select_action(self, state, env=None, do_apply_shield=True):
        self.last_obs = state
        context = context_provider.build_context(env or self.env, self)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits, value = self.policy(state_tensor)
        raw_probs = torch.softmax(logits, dim=-1)

        if do_apply_shield and self.shield_controller:
            shielded_probs = self.shield_controller.apply(raw_probs, context, self.verbose)
        else:
            shielded_probs = raw_probs

        # Check if shield changed any of the probabilities
        was_modified = not torch.allclose(raw_probs, shielded_probs, atol=1e-6)

        dist = Categorical(probs=shielded_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        if self.verbose:
            print(f"[Policy] Raw probs:       {raw_probs.detach().numpy().flatten()}")
            print(f"[Policy] Shielded probs:   {shielded_probs.detach().numpy().flatten()}")
            print(f"[Policy] Action selected:  {action.item()}")

        self.last_log_prob = log_prob.item()
        self.last_value = value.item()
        self.last_raw_probs = raw_probs.squeeze(0).detach()
        self.last_shielded_probs = shielded_probs.squeeze(0).detach()

        return action.item(), context, was_modified

    def store_transition(self, state, action, reward, next_state, context, done):
        self.memory.append((
            state, action, reward, next_state, context, done,
            self.last_log_prob, self.last_value,
            self.last_raw_probs, self.last_shielded_probs
        ))

    def update(self, batch_size=64, epochs=4):
        if len(self.memory) < batch_size:
            return

        self.learn_step_counter += 1

        states, actions, rewards, next_states, contexts, dones, log_probs, values, raw_probs, shielded_probs = zip(*self.memory)
        returns, advantages = self._compute_gae(rewards, values, dones)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        returns = torch.FloatTensor(np.array(returns))
        advantages = torch.FloatTensor(np.array(advantages))
        old_log_probs = torch.FloatTensor(np.array(log_probs))
        raw_probs = torch.stack(raw_probs)
        shielded_probs = torch.stack(shielded_probs)

        for _ in range(epochs):
            logits, new_values = self.policy(states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(new_values.squeeze(), returns)

            goal = torch.zeros_like(shielded_probs)
            goal.scatter_(1, actions.unsqueeze(1), 1.0)
            req_loss = nn.BCELoss()(shielded_probs, goal)
            consistency_loss = nn.MSELoss()(torch.softmax(logits, dim=-1), shielded_probs)

            loss = (policy_loss + 0.5 * value_loss - self.ent_coef * entropy +
                    self.lambda_req * req_loss + self.lambda_consistency * consistency_loss)

            # focal loss, similar approach with reinforcement learning?
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.verbose:
                print(f"[Update] Policy Loss:      {policy_loss.item():.4f}")
                print(f"[Update] Value Loss:       {value_loss.item():.4f}")
                print(f"[Update] Entropy:          {entropy.item():.4f}")
                print(f"[Update] Req Loss:         {req_loss.item():.4f}")
                print(f"[Update] Consistency Loss: {consistency_loss.item():.4f}")
                print(f"[Update] Total Loss:       {loss.item():.4f}")

        prob_shift = torch.abs(shielded_probs - raw_probs).mean().item()
        self.training_logs["policy_loss"].append(policy_loss.item())
        self.training_logs["value_loss"].append(value_loss.item())
        self.training_logs["entropy"].append(entropy.item())
        self.training_logs["req_loss"].append(req_loss.item())
        self.training_logs["consistency_loss"].append(consistency_loss.item())
        self.training_logs["prob_shift"].append(prob_shift)

        self.memory.clear()

    def _compute_gae(self, rewards, values, dones, lam=0.95):
        values = list(values) + [0]
        gae = 0
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * lam * (1 - dones[i]) * gae
            returns.insert(0, gae + values[i])
        advantages = [r - v for r, v in zip(returns, values[:-1])]
        return returns, advantages
