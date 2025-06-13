import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from src.models.network import QNetwork
from pishield.shield_layer import build_shield_layer

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
                 verbose=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.use_shield = use_shield
        self.verbose = verbose

        self.q_network = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is not trained directly

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64

        self.learn_step_counter = 0
        self.target_update_freq = target_update_freq

        if self.use_shield:
            self.shield_layer = build_shield_layer(
                action_dim,
                "src/requirements/constraints.linear",
                ordering_choice='given'
            )
        else:
            self.shield_layer = None

    def select_action(self, state, env=None):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_dim)
            if self.verbose:
                print(f"[Random] Action selected: {action}")
            return action

        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)  # (1, action_dim)
            action_probs = torch.softmax(q_values, dim=1)  # softmax for PiShield

            if self.use_shield and self.shield_layer is not None:
                corrected_probs = self.shield_layer(action_probs)
            else:
                corrected_probs = action_probs

            action = corrected_probs.argmax(dim=1).item()

            if self.verbose:
                print(f"[Policy] Q-values: {q_values.numpy().flatten()}")
                print(f"[Policy] Softmax probs: {action_probs.numpy().flatten()}")
                if self.use_shield:
                    print(f"[Policy] Shielded probs: {corrected_probs.numpy().flatten()}")
                print(f"[Policy] Action selected: {action}")

        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Epsilon decay
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def enable_shield(self, enable: bool):
        self.use_shield = enable
