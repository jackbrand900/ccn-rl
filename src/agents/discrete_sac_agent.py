import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from collections import deque

from src.models.network import ModularNetwork
from src.utils.shield_controller import ShieldController
from src.utils.constraint_monitor import ConstraintMonitor
from src.utils.preprocessing import prepare_input, prepare_batch
import src.utils.context_provider as context_provider


class DiscreteSACAgent:
    def __init__(self,
                 input_shape,
                 action_dim,
                 hidden_dim=256,
                 gamma=0.99,
                 alpha=0.2,
                 lr=3e-4,
                 buffer_size=100000,
                 batch_size=64,
                 tau=0.005,
                 use_cnn=False,
                 use_shield_post=False,
                 use_shield_layer=False,
                 monitor_constraints=True,
                 requirements_path=None,
                 mode='hard',
                 env=None,
                 verbose=False):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DiscreteSACAgent] Using device: {self.device}")
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.use_cnn = use_cnn
        self.use_shield_post = use_shield_post
        self.use_shield_layer = use_shield_layer
        self.monitor_constraints = monitor_constraints
        self.verbose = verbose
        self.env = env

        self.shield_controller = ShieldController(requirements_path, action_dim, mode, verbose)
        self.constraint_monitor = ConstraintMonitor(verbose=verbose)
        self.shield_controller.constraint_monitor = self.constraint_monitor

        self.q_net1 = ModularNetwork(input_shape, action_dim, hidden_dim, use_cnn=use_cnn).to(self.device)
        self.q_net2 = ModularNetwork(input_shape, action_dim, hidden_dim, use_cnn=use_cnn).to(self.device)
        self.target_q_net1 = ModularNetwork(input_shape, action_dim, hidden_dim, use_cnn=use_cnn).to(self.device)
        self.target_q_net2 = ModularNetwork(input_shape, action_dim, hidden_dim, use_cnn=use_cnn).to(self.device)
        self.policy_net = ModularNetwork(input_shape, action_dim, hidden_dim, use_cnn=use_cnn).to(self.device)

        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.replay_buffer = deque(maxlen=buffer_size)

    def select_action(self, state, env=None, do_apply_shield=True):
        self.last_obs = state
        context = context_provider.build_context(env or self.env, self)
        state_tensor = prepare_input(state, use_cnn=self.use_cnn).to(self.device)

        logits = self.policy_net(state_tensor, context=context)
        raw_probs = F.softmax(logits, dim=-1)
        shielded_probs = raw_probs.clone()

        was_shield_applied = False
        if self.use_shield_layer and do_apply_shield:
            shielded_probs = self.shield_controller.forward_differentiable(raw_probs, [context])[0]
            was_shield_applied = True
        elif self.use_shield_post and do_apply_shield:
            shielded_probs = self.shield_controller.apply(raw_probs, context)
            shielded_probs = shielded_probs[0] if shielded_probs.ndim == 2 else shielded_probs
            shielded_probs /= shielded_probs.sum()
            was_shield_applied = True

        dist = Categorical(probs=shielded_probs)
        action = dist.sample().item()

        if self.monitor_constraints:
            self.constraint_monitor.log_step(
                raw_probs=raw_probs.detach(),
                corrected_probs=shielded_probs.detach(),
                selected_action=action,
                shield_controller=self.shield_controller,
                context=context,
                shield_applied=was_shield_applied
            )

        return action, context

    def store_transition(self, state, action, reward, next_state, context, done):
        self.replay_buffer.append((state, action, reward, next_state, context, done))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, raw_contexts, dones = zip(*batch)

        states = prepare_batch(states, use_cnn=self.use_cnn).to(self.device)
        next_states = prepare_batch(next_states, use_cnn=self.use_cnn).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # === Normalize contexts ===
        contexts = []
        for ctx in raw_contexts:
            if isinstance(ctx, dict):
                contexts.append(ctx)
            elif isinstance(ctx, tuple):
                contexts.append({"obs": ctx[0], "direction": ctx[1]})
            else:
                raise ValueError(f"Invalid context format: {type(ctx)}")

        # === Target Q update ===
        with torch.no_grad():
            next_logits = self.policy_net(next_states, context=None)
            next_probs = F.softmax(next_logits, dim=-1)
            log_next_probs = F.log_softmax(next_logits, dim=-1)

            target_q1 = self.target_q_net1(next_states)
            target_q2 = self.target_q_net2(next_states)
            min_q = torch.min(target_q1, target_q2)

            entropy_term = self.alpha * log_next_probs
            target_v = (next_probs * (min_q - entropy_term)).sum(dim=-1)
            target_q = rewards + (1 - dones) * self.gamma * target_v

        q1 = self.q_net1(states).gather(1, actions.unsqueeze(1)).squeeze()
        q2 = self.q_net2(states).gather(1, actions.unsqueeze(1)).squeeze()

        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # === Policy update using shielded probs (Fix A) ===
        logits = self.policy_net(states)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            if self.use_shield_layer:
                shielded_probs = self.shield_controller.forward_differentiable(probs, contexts)
            elif self.use_shield_post:
                shielded_probs = []
                for p, c in zip(probs, contexts):
                    corrected = self.shield_controller.apply(p, c)
                    corrected /= corrected.sum()
                    shielded_probs.append(corrected)
                shielded_probs = torch.stack(shielded_probs).to(self.device)
            else:
                shielded_probs = probs.clone()

            q1 = self.q_net1(states)
            q2 = self.q_net2(states)
            min_q = torch.min(q1, q2)

        policy_loss = (shielded_probs * (self.alpha * log_probs - min_q)).sum(dim=1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # === Soft target updates ===
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def get_weights(self):
        return {
            "policy": self.policy_net.state_dict(),
            "q1": self.q_net1.state_dict(),
            "q2": self.q_net2.state_dict(),
        }

    def load_weights(self, weights):
        self.policy_net.load_state_dict(weights["policy"])
        self.q_net1.load_state_dict(weights["q1"])
        self.q_net2.load_state_dict(weights["q2"])
