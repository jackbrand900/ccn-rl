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


class VanillaDQNAgent:
    def __init__(self,
                 input_shape,
                 action_dim,
                 hidden_dim=128,
                 use_cnn=False,
                 gamma=0.99,
                 lr=1e-3,
                 batch_size=64,
                 buffer_size=100_000,
                 target_update_freq=500,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=10000,
                 use_shield_post=False,
                 use_shield_layer=False,
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

        # Shield setup
        self.constraint_monitor = ConstraintMonitor(verbose=verbose)
        self.shield_controller = ShieldController(
            requirements_path=requirements_path,
            num_actions=action_dim,
            mode=mode,
            verbose=verbose
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

        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       np.exp(-1.0 * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if deterministic or random.random() > self.epsilon:
            logits = self.q_net(state_tensor, context=context)
            raw_probs = torch.softmax(logits, dim=-1)
            shielded_probs = raw_probs.clone()

            if self.use_shield_layer and do_apply_shield:
                shielded_probs = self.shield_controller.forward_differentiable(raw_probs, [context]).squeeze(0)
                self.constraint_monitor.log_step(
                    raw_probs=raw_probs.detach(),
                    corrected_probs=shielded_probs.detach(),
                    selected_action=shielded_probs.argmax().item(),
                    shield_controller=self.shield_controller,
                    context=context
                )

            elif self.use_shield_post and do_apply_shield:
                shielded_probs = self.shield_controller.apply(raw_probs, context).squeeze(0)
                shielded_probs /= shielded_probs.sum()

                if self.constraint_monitor:
                    self.constraint_monitor.log_step(
                        raw_probs=raw_probs.detach(),
                        corrected_probs=shielded_probs.detach(),
                        selected_action=shielded_probs.argmax().item(),
                        shield_controller=self.shield_controller,
                        context=context
                    )

            dist = torch.distributions.Categorical(probs=shielded_probs)
            action = dist.sample().item()
        else:
            action = random.randrange(self.action_dim)

        return action, context

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
