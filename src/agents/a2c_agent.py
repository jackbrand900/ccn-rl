import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.models.network import ModularNetwork
from src.utils.preprocessing import prepare_input, prepare_batch
from src.utils.shield_controller import ShieldController
from src.utils.constraint_monitor import ConstraintMonitor
import src.utils.context_provider as context_provider


class A2CAgent:
    def __init__(self,
                 input_shape,
                 action_dim,
                 hidden_dim=128,
                 gamma=0.99,
                 lr=1e-3,
                 env=None,
                 entropy_coef=0.01,
                 use_cnn=False,
                 use_shield_post=False,
                 use_shield_layer=False,
                 monitor_constraints=False,
                 requirements_path=None,
                 mode='hard',
                 verbose=False):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[VanillaA2CAgent] Using device: {self.device}")

        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.use_cnn = use_cnn
        self.use_shield_post = use_shield_post
        self.use_shield_layer = use_shield_layer
        self.monitor_constraints = monitor_constraints
        self.verbose = verbose
        self.env = env
        self.action_dim = action_dim

        self.constraint_monitor = ConstraintMonitor(verbose=self.verbose)
        self.shield_controller = ShieldController(
            requirements_path=requirements_path,
            num_actions=action_dim,
            mode=mode,
            verbose=verbose,
            is_shield_active=(use_shield_layer or use_shield_post)
        )
        self.shield_controller.constraint_monitor = self.constraint_monitor

        self.model = ModularNetwork(
            input_shape=input_shape,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
            use_cnn=use_cnn,
            actor_critic=True,
            use_shield_layer=use_shield_layer,
            shield_controller=self.shield_controller if use_shield_layer else None
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = []

    def select_action(self, state, env=None, do_apply_shield=True):
        self.last_obs = state
        context = context_provider.build_context(env or self.env, self)
        state_tensor = prepare_input(state, use_cnn=self.use_cnn).to(self.device)

        logits, value = self.model(state_tensor, context=context)
        raw_probs = torch.softmax(logits, dim=-1)

        if self.use_shield_layer and do_apply_shield:
            shielded_probs = self.shield_controller.forward_differentiable(raw_probs, [context]).squeeze(0)
        elif self.use_shield_post and do_apply_shield:
            shielded_probs = self.shield_controller.apply(raw_probs, context).squeeze(0)
            shielded_probs /= shielded_probs.sum()
        else:
            shielded_probs = raw_probs.clone()

        # For monitoring
        dist_unshielded = Categorical(probs=raw_probs)
        a_unshielded = dist_unshielded.sample().item()

        dist_shielded = Categorical(probs=shielded_probs)
        a_shielded = dist_shielded.sample().item()

        selected_action = a_shielded if self.shield_controller.is_shield_active else a_unshielded
        dist_used = dist_shielded if self.shield_controller.is_shield_active else dist_unshielded
        log_prob = dist_used.log_prob(torch.tensor(selected_action).to(self.device))

        self.last_log_prob = log_prob
        self.last_value = value
        self.last_raw_probs = raw_probs.detach()
        self.last_shielded_probs = shielded_probs.detach()

        if self.monitor_constraints:
            self.constraint_monitor.log_step_from_probs_and_actions(
                raw_probs=raw_probs.detach(),
                corrected_probs=shielded_probs.detach(),
                a_unshielded=a_unshielded,
                a_shielded=a_shielded,
                context=context,
                shield_controller=self.shield_controller,
            )

        return selected_action, context

    def store_transition(self, state, action, reward, next_state, context, done):
        self.memory.append((
            prepare_input(state, use_cnn=self.use_cnn).squeeze(0).to(self.device),
            action,
            reward,
            prepare_input(next_state, use_cnn=self.use_cnn).squeeze(0).to(self.device),
            context,
            done,
            self.last_log_prob,
            self.last_value
        ))

    def update(self):
        if not self.memory:
            return

        # Unpack memory
        states, actions, rewards, next_states, contexts, dones, log_probs, values = zip(*self.memory)

        # Convert to tensors
        states = prepare_batch(states, use_cnn=self.use_cnn).to(self.device)
        next_states = prepare_batch(next_states, use_cnn=self.use_cnn).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)
        values = torch.stack(values).squeeze().to(self.device)

        # Ensure contexts are dictionaries
        contexts = [context_provider.build_context(self.env, self) if c is None else c for c in contexts]

        # Compute value targets and advantages
        with torch.no_grad():
            _, next_values = self.model(next_states)
            next_values = next_values.squeeze()
            targets = rewards + self.gamma * next_values * (1 - dones)
            advantages = targets - values

        # Forward pass on current states
        logits, predicted_values = self.model(states)
        raw_probs = torch.softmax(logits, dim=-1)

        # Apply shield if enabled
        if self.use_shield_layer:
            shielded_probs = self.shield_controller.forward_differentiable(raw_probs, contexts)
        elif self.use_shield_post:
            shielded_probs = torch.stack([
                self.shield_controller.apply(p.unsqueeze(0), c).squeeze(0)
                for p, c in zip(raw_probs, contexts)
            ])
            shielded_probs = shielded_probs / shielded_probs.sum(dim=1, keepdim=True)
        else:
            shielded_probs = raw_probs

        # Build Categorical dist and compute losses
        dist = Categorical(probs=shielded_probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        policy_loss = -(advantages.detach() * new_log_probs).mean()
        value_loss = nn.MSELoss()(predicted_values.squeeze(), targets)
        loss = policy_loss + value_loss - self.entropy_coef * entropy

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory.clear()

    def get_weights(self):
        return self.model.state_dict()

    def load_weights(self, weights):
        self.model.load_state_dict(weights)
