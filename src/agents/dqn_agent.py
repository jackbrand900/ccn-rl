import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from src.models.network import QNetwork
from pishield.shield_layer import build_shield_layer
from src.utils.shield_controller import ShieldController
from src.utils.env_helpers import extract_agent_pos, convert_action_to_string

class DQNAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim=64,
                 lr=1e-3,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_decay=0.995,
                 epsilon_min=0.01,
                 target_update_freq=1000,
                 use_shield=True,
                 verbose=False,
                 requirements_path=None,
                 env=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.use_shield = use_shield
        self.verbose = verbose
        self.env = env

        self.q_network = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is not trained directly
        self.last_raw_probs = None
        self.last_corrected_probs = None

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64

        self.learn_step_counter = 0
        self.target_update_freq = target_update_freq

        if self.use_shield:
            self.shield_controller = ShieldController(requirements_path, action_dim, None)
            self.shield_layer = self.shield_controller.build_shield_layer()
        else:
            self.shield_layer = None

    def select_action(self, state, env=None):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_dim)
            if self.verbose:
                print(f"[Random] Action selected: {action}")
            # Fallback: use uniform probs as dummy values
            uniform_probs = np.ones(self.action_dim, dtype=np.float32) / self.action_dim
            self.last_raw_probs = uniform_probs
            self.last_corrected_probs = uniform_probs
            return action

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action_probs = torch.softmax(q_values, dim=1)

            if self.use_shield and self.shield_layer is not None:
                position = extract_agent_pos(env)
                context = {
                    "state": state_tensor,
                    "position": position,
                    "step": self.learn_step_counter,
                    "env_info": getattr(env, "metadata", {})
                }
                corrected_probs = self.shield_controller.apply(action_probs, context, self.verbose)
            else:
                corrected_probs = action_probs

            action = corrected_probs.argmax(dim=1).item()

        self.last_raw_probs = action_probs.squeeze().cpu().numpy().astype(np.float32)
        self.last_corrected_probs = corrected_probs.squeeze().cpu().numpy().astype(np.float32)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        raw = self.last_raw_probs.astype(np.float32)
        corrected = self.last_corrected_probs.astype(np.float32)
        self.replay_buffer.append((state, action, reward, next_state, done, raw, corrected))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones, raw_probs, corrected_probs = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # --- Standard TD loss ---
        q_values_raw = self.q_network(states)
        q_values = q_values_raw.gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        td_loss = self.loss_fn(q_values, target_q_values)

        # --- Requirements loss ---
        raw_probs = np.array(raw_probs, dtype=np.float32)
        corrected_probs = np.array(corrected_probs, dtype=np.float32)

        raw_probs = torch.FloatTensor(raw_probs)
        corrected_probs = torch.FloatTensor(corrected_probs)

        action_mask = torch.zeros_like(corrected_probs).scatter_(1, actions, 1.0)
        log_probs = torch.log(raw_probs + 1e-8)
        selected_log_probs = (action_mask * log_probs).sum(dim=1, keepdim=True)
        with torch.no_grad():
            td_weights = target_q_values / (target_q_values.max() + 1e-8)

        req_loss = -selected_log_probs * td_weights
        req_loss = req_loss.mean()

        lambda_td = 1.0
        lambda_req = 0.05
        total_loss = lambda_td * td_loss + lambda_req * req_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        if self.verbose:
            print(f"[Update] TD Loss: {td_loss.item():.4f}, Req Loss: {req_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")

        # Update target network periodically
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Epsilon decay
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def enable_shield(self, enable: bool):
        self.use_shield = enable

    def get_agent_pos(self):
        return extract_agent_pos(self.env)
