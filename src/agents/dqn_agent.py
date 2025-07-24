from collections import deque
import numpy as np
from torch import optim
import torch
import random

from src.models.network import ModularNetwork
from src.utils import context_provider
from src.utils.shield_controller import ShieldController
from src.utils.constraint_monitor import ConstraintMonitor
from src.utils.preprocessing import prepare_input, prepare_batch


class DQNAgent:
    def __init__(self,
                 input_shape,
                 action_dim,
                 hidden_dim=64,
                 use_cnn=False,
                 lr=1e-3,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_decay=0.995,
                 epsilon_min=0.01,
                 target_update_freq=1000,
                 use_shield_post=True,
                 use_shield_layer=False,
                 verbose=False,
                 requirements_path=None,
                 env=None,
                 mode='hard'):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQNAgent] Using device: {self.device}")

        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.use_shield_post = use_shield_post
        self.use_shield_layer = use_shield_layer
        self.verbose = verbose
        self.env = env
        self.learn_step_counter = 0
        self.batch_size = 64

        # Init monitor + shield controller
        self.constraint_monitor = ConstraintMonitor(verbose=self.verbose)
        self.shield_controller = ShieldController(requirements_path, action_dim, mode)
        self.shield_controller.constraint_monitor = self.constraint_monitor

        self.q_network = ModularNetwork(
            input_shape=input_shape,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
            use_cnn=use_cnn,
            actor_critic=False,
            use_shield_layer=self.use_shield_layer,
            shield_controller=self.shield_controller
        ).to(self.device)

        self.target_network = ModularNetwork(
            input_shape=input_shape,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
            use_cnn=use_cnn,
            actor_critic=False
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=10000)
        self.target_update_freq = target_update_freq

        self.training_logs = {
            "td_loss": [], "req_loss": [], "consistency_loss": [], "prob_shift": []
        }

    def select_action(self, state, env=None, do_apply_shield=True):
        self.last_obs = state
        context = context_provider.build_context(env or self.env, self)
        state_tensor = prepare_input(state, use_cnn=self.q_network.use_cnn).to(self.device)

        # Use ModularNetwork's action selection if using the shield layer
        if self.use_shield_layer:
            action, _, _, raw_probs, shielded_probs = self.q_network.select_action(
                state_tensor, context=context, constraint_monitor=self.constraint_monitor, deterministic=True
            )

        else:
            if np.random.rand() < self.epsilon:
                action = np.random.choice(self.action_dim)
                if self.verbose:
                    print(f"[Random] Action selected: {action}")
                return action, context, False

            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                raw_probs = torch.softmax(q_values, dim=1)

                if do_apply_shield and self.use_shield_post:
                    shielded_probs = self.shield_controller.apply(raw_probs, context)

                    # Log if using post hoc
                    self.constraint_monitor.log_step(
                        raw_probs=raw_probs,
                        corrected_probs=shielded_probs,
                        selected_action=shielded_probs.argmax().item(),
                        shield_controller=self.shield_controller,
                        context=context
                    )
                else:
                    shielded_probs = raw_probs

                action = shielded_probs.argmax(dim=1).item()

                if self.verbose:
                    print(f"[Policy] Q-values: {q_values.cpu().numpy().flatten()}")
                    print(f"[Policy] Raw probs: {raw_probs.cpu().numpy().flatten()}")
                    print(f"[Policy] Shielded probs: {shielded_probs.cpu().numpy().flatten()}")
                    print(f"[Policy] Action selected: {action}")

        return action, context


    def store_transition(self, state, action, reward, next_state, context, done):
        if self.q_network.use_cnn and state.ndim == 1:
            state = state.reshape(96, 96, 3)
            next_state = next_state.reshape(96, 96, 3)
        self.replay_buffer.append((state, action, reward, next_state, context, done))

    def update(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        if len(self.replay_buffer) < batch_size:
            return

        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, contexts, dones = zip(*batch)

        states = prepare_batch(states, use_cnn=self.q_network.use_cnn).to(self.device)
        next_states = prepare_batch(next_states, use_cnn=self.q_network.use_cnn).to(self.device)

        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            raw_probs = torch.softmax(self.q_network(states), dim=1)
            if self.use_shield_post and self.shield_controller:
                shielded_probs = self.shield_controller.apply_batch(raw_probs, list(contexts))
            else:
                shielded_probs = raw_probs

        total_loss, logs = self.q_network.compute_q_loss(
            states, actions, rewards, dones, next_states, self.target_network,
            gamma=self.gamma, shielded_probs=shielded_probs
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        if self.verbose:
            print({k: f"{v:.4f}" for k, v in logs.items()})

        for k in self.training_logs:
            self.training_logs[k].append(logs.get(k, 0.0))

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def enable_shield(self, enable: bool):
        self.use_shield_post = enable

    def get_weights(self):
        return self.q_network.state_dict()

    def load_weights(self, weights):
        self.q_network.load_state_dict(weights)
