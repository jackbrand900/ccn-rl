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
                 agent_kwargs=None,
                 use_shield_post=False,
                 use_shield_layer=False,
                 monitor_constraints=True,
                 requirements_path=None,
                 use_cnn=False,
                 env=None,
                 verbose=False,
                 mode='hard'):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ShieldedDQNAgent] Using device: {self.device}")
        self.env = env

        self.gamma = 0.999
        self.lr = 1e-4
        self.batch_size = 64
        self.buffer_size = 100_000
        self.target_update_freq = 500
        self.epsilon_start = 0.5
        self.epsilon_end = 0.01
        self.epsilon_decay = 50000 # fix this
        self.hidden_dim = 128
        self.lambda_sem = 0.1
        self.num_layers = 3
        self.use_cnn = use_cnn
        self.use_orthogonal_init = True
        self.pretrained_cnn = None
        print(agent_kwargs)

        # === Override from agent_kwargs ===
        if agent_kwargs is not None:
            self.gamma = agent_kwargs.get("gamma", self.gamma)
            self.lr = agent_kwargs.get("lr", self.lr)
            self.batch_size = agent_kwargs.get("batch_size", self.batch_size)
            self.buffer_size = agent_kwargs.get("buffer_size", self.buffer_size)
            self.target_update_freq = agent_kwargs.get("target_update_freq", self.target_update_freq)
            self.epsilon_start = agent_kwargs.get("epsilon_start", self.epsilon_start)
            self.epsilon_end = agent_kwargs.get("epsilon_end", self.epsilon_end)
            self.epsilon_decay = agent_kwargs.get("epsilon_decay", self.epsilon_decay)
            self.hidden_dim = agent_kwargs.get("hidden_dim", self.hidden_dim)
            self.num_layers = agent_kwargs.get("num_layers", self.num_layers)
            self.use_cnn = agent_kwargs.get("use_cnn", self.use_cnn)
            self.use_orthogonal_init = agent_kwargs.get("use_orthogonal_init", self.use_orthogonal_init)
            self.pretrained_cnn = agent_kwargs.get("pretrained_cnn", self.pretrained_cnn)

        self.action_dim = action_dim
        self.use_shield_post = use_shield_post
        self.use_shield_layer = use_shield_layer
        self.monitor_constraints = monitor_constraints

        # === Shield ===
        self.constraint_monitor = ConstraintMonitor(verbose=verbose)
        self.shield_controller = ShieldController(
            requirements_path=requirements_path,
            num_actions=action_dim,
            mode=mode,
            verbose=verbose,
            is_shield_active=(self.use_shield_layer or self.use_shield_post)
        )
        self.shield_controller.constraint_monitor = self.constraint_monitor

        # === Networks ===
        self.q_net = ModularNetwork(
            input_shape=input_shape,
            output_dim=action_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            use_cnn=self.use_cnn,
            actor_critic=False,
            use_shield_layer=self.use_shield_layer,
            shield_controller=self.shield_controller,
            use_orthogonal_init=self.use_orthogonal_init,
            pretrained_cnn=self.pretrained_cnn
        ).to(self.device)

        self.target_net = ModularNetwork(
            input_shape=input_shape,
            output_dim=action_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            use_cnn=self.use_cnn,
            actor_critic=False,
            use_shield_layer=False,
            shield_controller=None,
            use_orthogonal_init=self.use_orthogonal_init,
            pretrained_cnn=self.pretrained_cnn
        ).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.replay_buffer = deque(maxlen=self.buffer_size)

        self.epsilon = self.epsilon_start
        self.steps_done = 0

        self.training_logs = {
            "loss": [],
            "epsilon": [],
            "prob_shift": [],
            "mod_rate": [],
            "td_loss": [],
            "semantic_loss": []
        }

    def select_action(self, state, deterministic=False, do_apply_shield=True):
        self.last_obs = state
        context = context_provider.build_context(self.env, self)
        state_tensor = prepare_input(state, use_cnn=self.use_cnn).to(self.device)

        # === Epsilon-greedy logic ===
        eps_diff = self.epsilon_start - self.epsilon_end
        decay_ratio = min(1.0, self.steps_done / self.epsilon_decay)
        self.epsilon = self.epsilon_start - decay_ratio * eps_diff
        self.steps_done += 1

        is_greedy = deterministic or random.random() > self.epsilon

        if is_greedy:
            logits = self.q_net(state_tensor, context=context)
            raw_probs = torch.softmax(logits, dim=-1)
            a_unshielded = torch.argmax(raw_probs).item()

            if self.use_shield_layer and do_apply_shield:
                corrected_probs = self.shield_controller.forward_differentiable(raw_probs, [context]).squeeze(0)
                a_shielded = torch.argmax(corrected_probs).item()
            elif self.use_shield_post and do_apply_shield:
                corrected_probs = self.shield_controller.apply(raw_probs, context).squeeze(0)
                corrected_probs /= corrected_probs.sum()
                a_shielded = torch.argmax(corrected_probs).item()
            else:
                corrected_probs = raw_probs.clone()
                a_shielded = a_unshielded

            selected_action = a_shielded if self.shield_controller.is_shield_active else a_unshielded
        else:
            selected_action = random.randrange(self.action_dim)
            raw_probs = torch.full((self.action_dim,), 1.0 / self.action_dim, device=self.device)
            corrected_probs = raw_probs.clone()
            a_unshielded = selected_action
            a_shielded = selected_action

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

        # === Q-Loss ===
        q_out = self.q_net(states, context=contexts)
        q_values = q_out.gather(1, actions)

        with torch.no_grad():
            next_q = self.target_net(next_states)
            next_max_q = next_q.max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * next_max_q * (1 - dones)

        td_loss = nn.MSELoss()(q_values, target_q_values)

        # === Semantic Loss (Xu et al. 2023) ===
        semantic_loss = 0
        if self.lambda_sem > 0:
            # Convert logits to probs
            with torch.no_grad():
                probs = torch.softmax(q_out, dim=-1)

            # Get logical flags
            flag_dicts = self.shield_controller.flag_logic_batch(contexts)
            flag_values = [
                [flags.get(name, 0.0) for name in self.shield_controller.flag_names]
                for flags in flag_dicts
            ]
            flag_tensor = torch.tensor(flag_values, dtype=probs.dtype, device=probs.device)

            # Concatenate for joint input to semantic loss
            probs_all = torch.cat([probs, flag_tensor], dim=1)
            semantic_loss = self.shield_controller.compute_semantic_loss(probs_all)

        total_loss = td_loss + self.lambda_sem * semantic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # === Logging ===
        self.training_logs["loss"].append(total_loss.item())
        self.training_logs["epsilon"].append(self.epsilon)
        self.training_logs["td_loss"].append(td_loss.item())
        if self.lambda_sem > 0:
            self.training_logs["semantic_loss"].append(semantic_loss.item())

    def get_weights(self):
        return self.q_net.state_dict()

    def load_weights(self, weights):
        self.q_net.load_state_dict(weights)
        self.target_net.load_state_dict(weights)
