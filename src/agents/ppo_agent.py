import random

import torch
from torch import nn

from src.utils.constraint_monitor import ConstraintMonitor
from src.utils.preprocessing import prepare_input, prepare_batch
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from src.models.network import ModularNetwork
from src.utils.shield_controller import ShieldController
import src.utils.context_provider as context_provider

class PPOAgent:
    def __init__(self,
                 input_shape,
                 action_dim,
                 hidden_dim=128,
                 use_cnn=False,
                 use_orthogonal_init=False,
                 lr=5e-4,
                 gamma=0.99,
                 clip_eps=0.2,
                 ent_coef=0.01,
                 lambda_sem=0.1,
                 lambda_consistency=0,
                 verbose=False,
                 requirements_path=None,
                 env=None,
                 batch_size=64,
                 epochs=4,
                 use_shield_post=False,
                 use_shield_pre=False,
                 use_shield_layer=False,
                 monitor_constraints=True,
                 agent_kwargs=None,
                 mode='hard'):

        self.lambda_sem = lambda_sem
        self.lambda_consistency = lambda_consistency
        self.use_shield_post = use_shield_post
        self.use_shield_pre = use_shield_pre
        self.use_shield_layer = use_shield_layer
        self.monitor_constraints = monitor_constraints
        self.verbose = verbose
        self.env = env
        self.action_dim = action_dim

        print(agent_kwargs)
        if agent_kwargs is not None:
            self.hidden_dim = agent_kwargs.get("hidden_dim", hidden_dim)
            self.use_orthogonal_init = agent_kwargs.get("use_orthogonal_init", use_orthogonal_init)
            self.lr = agent_kwargs.get("lr", lr)
            self.gamma = agent_kwargs.get("gamma", gamma)
            self.clip_eps = agent_kwargs.get("clip_eps", clip_eps)
            self.ent_coef = agent_kwargs.get("ent_coef", ent_coef)
            self.batch_size = agent_kwargs.get("batch_size", batch_size)
            self.epochs = agent_kwargs.get("epochs", epochs)
            self.num_layers = agent_kwargs.get("num_layers", 2)
        else:
            self.lr = lr
            self.gamma = gamma
            self.clip_eps = clip_eps
            self.ent_coef = ent_coef
            self.batch_size = batch_size
            self.epochs = epochs
            self.use_orthogonal_init = use_orthogonal_init
            self.hidden_dim = hidden_dim
            self.num_layers = 3

        self.use_cnn = use_cnn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = []

        self.learn_step_counter = 0
        self.last_log_prob = None
        self.last_value = None
        self.last_raw_probs = None
        self.last_shielded_probs = None
        self.last_obs = None
        is_shield_active = self.use_shield_layer or self.use_shield_post or self.use_shield_pre
        self.shield_controller = ShieldController(requirements_path, action_dim, mode, verbose=self.verbose, is_shield_active=is_shield_active)
        self.policy = ModularNetwork(input_shape, action_dim, self.hidden_dim, num_layers=self.num_layers,
                                     use_shield_layer=self.use_shield_layer, pretrained_cnn=None,
                                     use_cnn=use_cnn, actor_critic=True, use_orthogonal_init=self.use_orthogonal_init,
                                     shield_controller=self.shield_controller).to(self.device)
        self.constraint_monitor = ConstraintMonitor(verbose=self.verbose)
        self.shield_controller.constraint_monitor = self.constraint_monitor
        print(f"[PPOAgent] Using device: {self.device}")
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)

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
        state_tensor = prepare_input(state, use_cnn=self.use_cnn).to(self.device)

        # Always get raw_probs from the policy
        action, log_prob, value, raw_probs, shielded_probs = self.policy.select_action(
            state_tensor, context,
            constraint_monitor=self.constraint_monitor if self.use_shield_layer else None
        )

        if self.use_shield_layer:
            # Action was selected from shielded_probs inside the model
            a_shielded = action
            # Sample a_unshielded from raw_probs for monitoring
            dist_unshielded = torch.distributions.Categorical(probs=raw_probs)
            a_unshielded = dist_unshielded.sample().item()
            selected_action = a_shielded
            log_prob_tensor = log_prob

        elif self.use_shield_post or self.use_shield_pre and do_apply_shield:
            # Sample unshielded action from raw_probs
            dist_unshielded = torch.distributions.Categorical(probs=raw_probs)
            a_unshielded = dist_unshielded.sample().item()

            # Apply post hoc shield and sample again
            shielded_probs = self.shield_controller.apply(raw_probs.unsqueeze(0), context).squeeze(0)
            shielded_probs /= shielded_probs.sum()
            dist_shielded = torch.distributions.Categorical(probs=shielded_probs)
            a_shielded = dist_shielded.sample().item()

            selected_action = a_shielded
            log_prob_tensor = dist_shielded.log_prob(torch.tensor(a_shielded).to(self.device))

        else:
            # === Unshielded path: sample raw and get hypothetical shielded action
            dist_unshielded = torch.distributions.Categorical(probs=raw_probs)
            a_unshielded = dist_unshielded.sample().item()
            log_prob_tensor = dist_unshielded.log_prob(torch.tensor(a_unshielded).to(self.device))

            if self.monitor_constraints:
                shielded_probs = self.shield_controller.apply(raw_probs.unsqueeze(0), context).squeeze(0)
                shielded_probs /= shielded_probs.sum()
                dist_shielded = torch.distributions.Categorical(probs=shielded_probs)
                a_shielded = dist_shielded.sample().item()

            selected_action = a_unshielded

        # Save for training
        self.last_log_prob = log_prob_tensor.item()
        self.last_value = value.item()
        self.last_raw_probs = raw_probs.detach()
        self.last_shielded_probs = shielded_probs.detach()

        # === Constraint monitoring ===
        if self.monitor_constraints:
            self.constraint_monitor.log_step_from_probs_and_actions(
                raw_probs=raw_probs.detach(),
                corrected_probs=shielded_probs.detach(),
                a_unshielded=a_unshielded,
                a_shielded=a_shielded,
                context=context,
                shield_controller=self.shield_controller,
            )

        return selected_action, a_unshielded, a_shielded, context

    def store_transition(self, state, action, reward, next_state, context, done):
        self.memory.append((
            state, action, reward, next_state, context, done,
            self.last_log_prob, self.last_value,
            self.last_raw_probs, self.last_shielded_probs
        ))

    def ensure_dict_contexts(self, contexts):
        new_contexts = []
        for c in contexts:
            if isinstance(c, dict):
                new_contexts.append(c)
            elif isinstance(c, tuple):
                new_contexts.append({"obs": c[0], "direction": c[1]})  # customize as needed
            else:
                raise ValueError(f"Invalid context type: {type(c)} - {c}")
        return new_contexts

    def update(self, batch_size=None, epochs=None):
        batch_size = batch_size or self.batch_size
        epochs = epochs or self.epochs
        if len(self.memory) < batch_size:
            return

        self.learn_step_counter += 1

        # === Unpack memory ===
        states, actions, rewards, next_states, contexts, dones, log_probs, values, raw_probs, shielded_probs = zip(*self.memory)
        contexts = self.ensure_dict_contexts(contexts)
        returns, advantages = self._compute_gae(rewards, values, dones)

        # === Prepare tensors ===
        states = prepare_batch(states, use_cnn=self.use_cnn).to(self.device)
        if self.use_cnn and states.ndim == 4 and states.shape[-1] == 3:
            states = states.permute(0, 3, 1, 2)

        actions = torch.LongTensor(actions).to(self.device)
        returns = torch.FloatTensor(np.array(returns)).to(self.device)
        advantages = torch.FloatTensor(np.array(advantages)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(log_probs)).to(self.device)
        raw_probs = torch.stack(raw_probs).to(self.device)
        # shielded_probs = torch.stack(shielded_probs).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(epochs):
            logits, predicted_values = self.policy(states, context=contexts)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(predicted_values.squeeze(), returns)

            probs = torch.softmax(logits, dim=-1)
            semantic_loss = 0
            if self.lambda_sem > 0:
                flag_dicts = self.shield_controller.flag_logic_batch(contexts)
                flag_values = [
                    [flags.get(name, 0.0) for name in self.shield_controller.flag_names]
                    for flags in flag_dicts
                ]
                flag_tensor = torch.tensor(flag_values, dtype=probs.dtype, device=probs.device)
                probs_all = torch.cat([probs, flag_tensor], dim=1)  # [B, num_vars]
                semantic_loss = self.shield_controller.compute_semantic_loss(probs_all)
            total_loss = (policy_loss +
                          0.5 * value_loss -
                          self.ent_coef * entropy +
                          self.lambda_sem * semantic_loss)

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

            logs = {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy.item(),
                "total_loss": total_loss.item()
            }

            for k in self.training_logs:
                self.training_logs[k].append(logs.get(k, 0.0))

        self.scheduler.step()
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

    def get_weights(self):
        return self.policy.state_dict()

    def load_weights(self, weights):
        self.policy.load_state_dict(weights)