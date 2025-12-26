import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.models.network import ModularNetwork
from src.utils.preprocessing import prepare_input, prepare_batch
from src.utils.shield_controller import ShieldController
from src.utils.constraint_monitor import ConstraintMonitor
import src.utils.context_provider as context_provider


class VanillaA2CAgent:
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
        self.verbose = verbose
        self.env = env

        # === Shielding Setup ===
        self.constraint_monitor = ConstraintMonitor(verbose=self.verbose)
        self.shield_controller = ShieldController(
            requirements_path=requirements_path,
            num_actions=action_dim,
            mode=mode,
            verbose=verbose
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
        context = context_provider.build_context(env or self.env, self)
        state_tensor = prepare_input(state, use_cnn=self.use_cnn).to(self.device)

        logits, value = self.model(state_tensor, context=context)
        raw_probs = torch.softmax(logits, dim=-1)
        shielded_probs = raw_probs.clone()

        if self.use_shield_layer and do_apply_shield:
            shielded_probs = self.shield_controller.forward_differentiable(raw_probs, [context]).squeeze(0)
            if self.constraint_monitor:
                self.constraint_monitor.log_step(
                    raw_probs=raw_probs.detach(),
                    corrected_probs=shielded_probs.detach(),
                    selected_action=shielded_probs.argmax().item(),
                    shield_controller=self.shield_controller,
                    context=context
                )

        elif self.use_shield_post and do_apply_shield:
            shielded_probs = self.shield_controller.apply(raw_probs, context).squeeze(0)
            shielded_probs = shielded_probs / shielded_probs.sum()
            if self.constraint_monitor:
                self.constraint_monitor.log_step(
                    raw_probs=raw_probs.detach(),
                    corrected_probs=shielded_probs.detach(),
                    selected_action=shielded_probs.argmax().item(),
                    shield_controller=self.shield_controller,
                    context=context
                )

        dist = Categorical(probs=shielded_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Save for later
        self.last_log_prob = log_prob
        self.last_value = value
        self.last_raw_probs = raw_probs.detach()
        self.last_shielded_probs = shielded_probs.detach()

        return action.item(), context

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

        states, actions, rewards, next_states, contexts, dones, log_probs, values = zip(*self.memory)

        states = prepare_batch(states, use_cnn=self.use_cnn).to(self.device)
        next_states = prepare_batch(next_states, use_cnn=self.use_cnn).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)
        values = torch.stack(values).squeeze().to(self.device)

        with torch.no_grad():
            _, next_values = self.model(next_states)
            next_values = next_values.squeeze()
            targets = rewards + self.gamma * next_values * (1 - dones)
            advantages = targets - values

        logits, predicted_values = self.model(states)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        policy_loss = -(advantages.detach() * new_log_probs).mean()
        value_loss = nn.MSELoss()(predicted_values.squeeze(), targets)
        loss = policy_loss + value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory.clear()

    def get_weights(self):
        return self.model.state_dict()

    def load_weights(self, weights):
        self.model.load_state_dict(weights)
