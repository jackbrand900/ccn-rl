import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.utils.preprocessing import prepare_input, prepare_batch
from src.models.network import ModularNetwork
from src.utils.shield_controller import ShieldController
import src.utils.context_provider as context_provider

class A2CAgent:
    def __init__(self,
                 input_shape,
                 action_dim,
                 hidden_dim=128,
                 lr=1e-3,
                 gamma=0.99,
                 use_cnn=False,
                 use_shield=True,
                 verbose=False,
                 requirements_path=None,
                 env=None,
                 mode='hard'):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[A2CAgent] Using device: {self.device}")

        self.gamma = gamma
        self.use_shield = use_shield
        self.verbose = verbose
        self.env = env
        self.learn_step_counter = 0

        self.model = ModularNetwork(input_shape=input_shape, output_dim=action_dim,
                                    hidden_dim=hidden_dim, use_cnn=use_cnn, actor_critic=True).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.memory = []
        self.training_logs = {"policy_loss": [], "value_loss": [], "entropy": [], "prob_shift": []}

        if use_shield:
            self.shield_controller = ShieldController(
                requirements_path=requirements_path,
                num_actions=action_dim,
                mode=mode,
            )
        else:
            self.shield_controller = None

    def select_action(self, state, env=None, do_apply_shield=True):
        context = context_provider.build_context(env or self.env, self)
        state_tensor = prepare_input(state, use_cnn=self.model.use_cnn).to(self.device)

        logits, value = self.model(state_tensor)
        raw_probs = torch.softmax(logits, dim=-1)

        if do_apply_shield and self.shield_controller:
            shielded_probs = self.shield_controller.apply(raw_probs, context, self.verbose)
        else:
            shielded_probs = raw_probs

        was_modified = not torch.allclose(raw_probs, shielded_probs, atol=1e-6)

        dist = Categorical(probs=shielded_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.last_log_prob = log_prob
        self.last_value = value.squeeze(0)
        self.last_raw_probs = raw_probs.squeeze(0).detach()
        self.last_shielded_probs = shielded_probs.squeeze(0).detach()

        return action.item(), context, was_modified

    def store_transition(self, state, action, reward, next_state, context, done):
        state = prepare_input(state, use_cnn=self.model.use_cnn).squeeze(0).to(self.device)
        next_state = prepare_input(next_state, use_cnn=self.model.use_cnn).squeeze(0).to(self.device)

        self.memory.append((state, action, reward, next_state, context, done,
                            self.last_log_prob, self.last_value,
                            self.last_raw_probs, self.last_shielded_probs))

    def update(self, batch_size=None):
        if not self.memory:
            return

        self.learn_step_counter += 1

        states, actions, rewards, next_states, contexts, dones, log_probs, values, raw_probs, shielded_probs = zip(*self.memory)

        states = prepare_batch(states, use_cnn=self.model.use_cnn).to(self.device)
        next_states = prepare_batch(next_states, use_cnn=self.model.use_cnn).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        values = torch.stack(values).to(self.device)
        raw_probs = torch.stack(raw_probs).to(self.device)
        shielded_probs = torch.stack(shielded_probs).to(self.device)

        with torch.no_grad():
            _, next_values = self.model(next_states)
            targets = rewards + self.gamma * next_values.squeeze() * (1 - dones)
            advantages = targets - values.squeeze()

        logits, predicted_values = self.model(states)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        policy_loss = -(advantages.detach() * new_log_probs).mean()
        value_loss = nn.MSELoss()(predicted_values.view(-1), targets.view(-1))

        goal = torch.zeros_like(shielded_probs)
        goal.scatter_(1, actions.unsqueeze(1), 1.0)
        req_loss = nn.BCELoss()(shielded_probs, goal)
        consistency_loss = nn.MSELoss()(torch.softmax(logits, dim=-1), shielded_probs)

        loss = policy_loss + value_loss + 0.00 * req_loss + 0.00 * consistency_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_logs["policy_loss"].append(policy_loss.item())
        self.training_logs["value_loss"].append(value_loss.item())
        self.training_logs["entropy"].append(entropy.item())
        self.training_logs["prob_shift"].append((shielded_probs - raw_probs).abs().mean().item())

        self.memory.clear()

    def get_weights(self):
        return self.model.state_dict()

    def load_weights(self, weights):
        self.model.load_state_dict(weights)
