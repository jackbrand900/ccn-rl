import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

from src.utils.preprocessing import prepare_input, prepare_batch
from src.models.network import ModularNetwork
from src.utils.shield_controller import ShieldController
from src.utils.constraint_monitor import ConstraintMonitor
import src.utils.context_provider as context_provider


class DQNAgent:
    def __init__(self,
                 input_shape,
                 action_dim,
                 hidden_dim=128,
                 use_cnn=False,
                 gamma=0.99,
                 lr=2e-4,
                 batch_size=64,
                 buffer_size=100_000,
                 target_update_freq=500,
                 epsilon_start=0.5,
                 epsilon_end=0.01,
                 epsilon_decay=10000,
                 use_shield_post=False,
                 use_shield_layer=False,
                 monitor_constraints=True,
                 requirements_path=None,
                 env=None,
                 verbose=False,
                 mode='hard'):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ShieldedDQNAgent] Using device: {self.device}")
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.steps_done = 0
        self.target_update_freq = target_update_freq
        self.action_dim = action_dim
        self.use_cnn = use_cnn
        self.use_shield_post = use_shield_post
        self.use_shield_layer = use_shield_layer
        self.monitor_constraints = monitor_constraints

        # Shield setup
        self.constraint_monitor = ConstraintMonitor(verbose=verbose)
        self.shield_controller = ShieldController(
            requirements_path=requirements_path,
            num_actions=action_dim,
            mode=mode,
            verbose=verbose,
            is_shield_active=(self.use_shield_layer or self.use_shield_post)
        )
        self.shield_controller.constraint_monitor = self.constraint_monitor

        self.q_net = ModularNetwork(
            input_shape=input_shape,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
            use_cnn=use_cnn,
            actor_critic=False,
            use_shield_layer=use_shield_layer,
            shield_controller=self.shield_controller
        ).to(self.device)

        self.target_net = ModularNetwork(
            input_shape=input_shape,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
            use_cnn=use_cnn,
            actor_critic=False,
            use_shield_layer=False,
            shield_controller=None  # target net doesn’t need shield
        ).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=buffer_size)

        self.training_logs = {
            "loss": [],
            "epsilon": [],
            "prob_shift": [],
            "mod_rate": []
        }

    def select_action(self, state, deterministic=False, do_apply_shield=True):
        self.last_obs = state
        context = context_provider.build_context(self.env, self)
        state_tensor = prepare_input(state, use_cnn=self.use_cnn).to(self.device)

        # === Epsilon-greedy logic ===
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       np.exp(-1.0 * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        is_greedy = deterministic or random.random() > self.epsilon

        if is_greedy:
            # === Compute raw probabilities ===
            logits = self.q_net(state_tensor, context=context)
            raw_probs = torch.softmax(logits, dim=-1)

            # === Get unshielded action ===
            dist_unshielded = torch.distributions.Categorical(probs=raw_probs)
            a_unshielded = dist_unshielded.sample().item()

            # === Apply shield (if enabled) ===
            if self.use_shield_layer and do_apply_shield:
                corrected_probs = self.shield_controller.forward_differentiable(raw_probs, [context]).squeeze(0)
            elif self.use_shield_post and do_apply_shield:
                corrected_probs = self.shield_controller.apply(raw_probs, context).squeeze(0)
                corrected_probs /= corrected_probs.sum()
            else:
                corrected_probs = raw_probs.clone()

            dist_shielded = torch.distributions.Categorical(probs=corrected_probs)
            a_shielded = dist_shielded.sample().item()

            selected_action = a_shielded if self.shield_controller.is_shield_active else a_unshielded

        else:
            # === Random action (epsilon) ===
            selected_action = random.randrange(self.action_dim)
            raw_probs = torch.full((self.action_dim,), 1.0 / self.action_dim, device=self.device)
            corrected_probs = raw_probs.clone()
            a_unshielded = selected_action
            a_shielded = selected_action  # treated same for logging

        # === Log constraints ===
        if self.monitor_constraints:
            self.constraint_monitor.log_step_from_probs_and_actions(
                raw_probs=raw_probs.detach(),
                corrected_probs=corrected_probs.detach(),
                a_unshielded=a_unshielded,
                a_shielded=a_shielded,
                context=context,
                shield_controller=self.shield_controller
            )

        return selected_action, context

    def store_transition(self, state, action, reward, next_state, context, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        contexts = [context_provider.build_context(self.env, self) for _ in range(len(states))]

        states = prepare_batch(states, use_cnn=self.use_cnn).to(self.device)
        next_states = prepare_batch(next_states, use_cnn=self.use_cnn).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        logits = self.q_net(states, context=contexts)
        q_values = logits.gather(1, actions)

        with torch.no_grad():
            next_logits = self.target_net(next_states)
            next_q = next_logits.max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Sync target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Logging
        raw_probs = torch.softmax(logits, dim=-1).detach()
        shielded_probs = raw_probs.clone()

        if self.use_shield_layer:
            shielded_probs = self.shield_controller.forward_differentiable(raw_probs, contexts).detach()

        prob_shift = torch.abs(raw_probs - shielded_probs).mean().item()
        mod_rate = (shielded_probs.argmax(dim=1) != raw_probs.argmax(dim=1)).float().mean().item()

        self.training_logs["loss"].append(loss.item())
        self.training_logs["epsilon"].append(self.epsilon)
        self.training_logs["prob_shift"].append(prob_shift)
        self.training_logs["mod_rate"].append(mod_rate)

    def get_weights(self):
        return self.q_net.state_dict()

    def load_weights(self, weights):
        self.q_net.load_state_dict(weights)
        self.target_net.load_state_dict(weights)
