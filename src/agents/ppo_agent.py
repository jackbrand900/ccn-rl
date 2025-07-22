import torch
from src.utils.preprocessing import prepare_input, prepare_batch
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

from src.models.network import ModularNetwork
from src.utils.shield_controller import ShieldController
import src.utils.context_provider as context_provider

class PPOAgent:
    def __init__(self, input_shape, action_dim, hidden_dim=256, use_cnn=False, lr=3e-4,
                 gamma=0.99, clip_eps=0.2, ent_coef=0.01, lambda_req=0.0,
                 lambda_consistency=0.0, use_shield=True, verbose=False,
                 requirements_path=None, env=None, mode='hard',
                 batch_size=4096, epochs=8):

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.lambda_req = lambda_req
        self.lambda_consistency = lambda_consistency
        self.use_shield = use_shield
        self.verbose = verbose
        self.env = env
        self.action_dim = action_dim
        self.use_cnn = use_cnn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ModularNetwork(input_shape, action_dim, hidden_dim,
                                     use_cnn=use_cnn, actor_critic=True).to(self.device)
        print(f"[PPOAgent] Using device: {self.device}")
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)
        self.memory = []

        self.batch_size = batch_size
        self.epochs = epochs

        self.learn_step_counter = 0
        self.last_log_prob = None
        self.last_value = None
        self.last_raw_probs = None
        self.last_shielded_probs = None
        self.last_obs = None

        self.shield_controller = ShieldController(requirements_path, action_dim, mode) if use_shield else None

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

        action, log_prob, value, raw_probs = self.policy.select_action(state_tensor)

        if do_apply_shield and self.shield_controller:
            shielded_probs = self.shield_controller.apply(raw_probs.unsqueeze(0), context, self.verbose).squeeze(0)
            shielded_probs /= shielded_probs.sum()
            was_modified = not torch.allclose(raw_probs, shielded_probs, atol=1e-6)

            dist = Categorical(probs=shielded_probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action).to(self.device))
        else:
            shielded_probs = raw_probs
            was_modified = False

        self.last_log_prob = log_prob.item()
        self.last_value = value.item()
        self.last_raw_probs = raw_probs.detach()
        self.last_shielded_probs = shielded_probs.detach()

        return action, context, was_modified

    def store_transition(self, state, action, reward, next_state, context, done):
        self.memory.append((
            state, action, reward, next_state, context, done,
            self.last_log_prob, self.last_value,
            self.last_raw_probs, self.last_shielded_probs
        ))

    def update(self, batch_size=None, epochs=None):
        batch_size = batch_size or self.batch_size
        epochs = epochs or self.epochs
        if len(self.memory) < batch_size:
            return

        self.learn_step_counter += 1

        states, actions, rewards, next_states, contexts, dones, log_probs, values, raw_probs, shielded_probs = zip(*self.memory)
        returns, advantages = self._compute_gae(rewards, values, dones)

        states = prepare_batch(states, use_cnn=self.use_cnn).to(self.device)
        if self.use_cnn and states.ndim == 4 and states.shape[-1] == 3:
            states = states.permute(0, 3, 1, 2)

        actions = torch.LongTensor(actions).to(self.device)
        returns = torch.FloatTensor(np.array(returns)).to(self.device)
        advantages = torch.FloatTensor(np.array(advantages)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(log_probs)).to(self.device)
        raw_probs = torch.stack(raw_probs).to(self.device)
        shielded_probs = torch.stack(shielded_probs).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(epochs):
            loss, logs = self.policy.compute_losses(
                states, actions, old_log_probs, advantages, returns, shielded_probs,
                clip_eps=self.clip_eps,
                ent_coef=self.ent_coef,
                lambda_req=self.lambda_req,
                lambda_consistency=self.lambda_consistency
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        self.scheduler.step()
        prob_shift = torch.abs(shielded_probs - raw_probs).mean().item()
        for k in self.training_logs:
            self.training_logs[k].append(logs.get(k, 0.0))
        self.training_logs["prob_shift"].append(prob_shift)

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
