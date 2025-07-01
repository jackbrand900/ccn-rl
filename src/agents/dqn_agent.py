import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from src.models.network import QNetwork
from src.utils.shield_controller import ShieldController
import src.utils.context_provider as context_provider

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
        self.target_network.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64
        self.learn_step_counter = 0
        self.target_update_freq = target_update_freq
        self.training_logs = {
            "td_loss": [],
            "req_loss": [],
            "consistency_loss": [],
            "prob_shift": []
        }

        if self.use_shield:
            self.shield_controller = ShieldController(
                requirements_path=requirements_path,
                num_actions=action_dim,
                flag_logic_fn=context_provider.position_flag_logic,
            )
        else:
            self.shield_controller = None

    def select_action(self, state, env=None):
        context = context_provider.build_context(env, self)
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_dim)
            if self.verbose:
                print(f"[Random] Action selected: {action}")
            return action, context, False  # no shield applied during random choice

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(next(self.q_network.parameters()).device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action_probs = torch.softmax(q_values, dim=1)

            if self.use_shield and self.shield_controller:
                original_action = action_probs.argmax(dim=1).item()
                corrected_probs = self.shield_controller.apply(action_probs, context, self.verbose)
                shielded_action = corrected_probs.argmax(dim=1).item()
                was_modified = shielded_action != original_action
            else:
                corrected_probs = action_probs[:, :self.action_dim]
                was_modified = False

            action = corrected_probs.argmax(dim=1).item()

            if self.verbose:
                print(f"[Policy] Q-values: {q_values.cpu().numpy().flatten()}")
                print(f"[Policy] Raw probs: {action_probs.cpu().numpy().flatten()}")
                print(f"[Policy] Shielded probs: {corrected_probs.cpu().numpy().flatten()}")
                print(f"[Policy] Action selected: {action}")
                print(f"[Policy] Shield modified: {was_modified}")

        return action, context, was_modified

    def store_transition(self, state, action, reward, next_state, context, done):
        self.replay_buffer.append((state, action, reward, next_state, context, done))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample and unpack
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, contexts, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # --- TD Loss ---
        q_out = self.q_network(states)
        q_values = q_out.gather(1, actions)

        with torch.no_grad():
            next_q = self.target_network(next_states)
            next_max_q = next_q.max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * next_max_q * (1 - dones)

        td_loss = self.loss_fn(q_values, target_q_values)

        # --- Requirements Loss ---
        raw_probs = torch.softmax(q_out, dim=1)

        if self.use_shield and self.shield_controller:
            shielded_probs = self.shield_controller.apply_batch(raw_probs, list(contexts))
        else:
            shielded_probs = raw_probs[:, :self.action_dim]

        goal = torch.zeros_like(shielded_probs)
        goal.scatter_(1, actions, 1.0)

        req_loss = nn.BCELoss()(shielded_probs, goal)

        # --- Consistency Loss (encourage raw probs ≈ shielded probs) ---
        consistency_loss = torch.nn.MSELoss()(raw_probs[:, :self.action_dim], shielded_probs)

        # --- Total Loss ---
        lambda_td = 1.0
        lambda_req = 0.05
        lambda_consistency = 0.05
        total_loss = (
                lambda_td * td_loss +
                lambda_req * req_loss +
                lambda_consistency * consistency_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        if self.verbose:
            print(f"[Update] TD Loss: {td_loss.item():.4f}, Req Loss: {req_loss.item():.4f}, Consistency: {consistency_loss.item():.4f}, Total: {total_loss.item():.4f}")


        # Log metrics
        self.training_logs["td_loss"].append(td_loss.item())
        self.training_logs["req_loss"].append(req_loss.item())
        self.training_logs["consistency_loss"].append(consistency_loss.item())

        # Compute average per-batch L1 shift in probability due to the shield
        prob_shift = torch.abs(shielded_probs - raw_probs[:, :self.action_dim]).mean().item()
        self.training_logs["prob_shift"].append(prob_shift)

        # Target network update
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Epsilon decay
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def enable_shield(self, enable: bool):
        self.use_shield = enable
