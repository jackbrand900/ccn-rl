import torch
import torch.nn as nn
from torch.distributions import Categorical


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.model(x)


class ModularNetwork(nn.Module):
    def __init__(self, input_shape, output_dim, hidden_dim=128, use_cnn=False, actor_critic=False):
        super().__init__()
        self.use_cnn = use_cnn
        self.actor_critic = actor_critic

        if use_cnn:
            if len(input_shape) == 3:
                if input_shape[-1] == 3:
                    h, w, c = input_shape
                else:
                    c, h, w = input_shape
            else:
                raise ValueError(f"Expected 3D input shape, got {input_shape}")

            self.encoder = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
            )

            with torch.no_grad():
                dummy = torch.zeros(1, c, h, w).float() / 255.0
                conv_out = self.encoder(dummy)
                norm_shape = conv_out.shape[1:]
                conv_out_dim = conv_out.reshape(1, -1).size(1)
                print(f"[DEBUG] conv_out shape: {conv_out.shape}, flattened: {conv_out_dim}")

            self.encoder.add_module("norm", nn.LayerNorm(norm_shape))

        else:
            self.encoder = nn.Identity()
            conv_out_dim = input_shape if isinstance(input_shape, int) else int(torch.tensor(input_shape).prod())

        if actor_critic:
            self.actor = nn.Sequential(
                nn.Linear(conv_out_dim, hidden_dim), nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_dim, output_dim)
            )
            self.critic = nn.Sequential(
                nn.Linear(conv_out_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.q_net = nn.Sequential(
                nn.Linear(conv_out_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x):
        if self.use_cnn:
            if x.ndim == 4 and x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)
            x = x.float() / 255.0
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)

        if self.actor_critic:
            return self.actor(x), self.critic(x)
        else:
            return self.q_net(x)

    def select_action(self, state_tensor):
        logits, value = self.forward(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value.squeeze(), probs.squeeze(0)

    def compute_losses(self, states, actions, old_log_probs, advantages, returns, shielded_probs,
                       clip_eps=0.2, ent_coef=0.01, lambda_req=0.0, lambda_consistency=0.0):
        logits, new_values = self.forward(states)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.MSELoss()(new_values.squeeze(), returns)

        goal = torch.zeros_like(shielded_probs)
        goal.scatter_(1, actions.unsqueeze(1), 1.0)
        req_loss = nn.BCELoss()(shielded_probs, goal)
        consistency_loss = nn.MSELoss()(torch.softmax(logits, dim=-1), shielded_probs)

        total_loss = (policy_loss +
                      0.5 * value_loss -
                      ent_coef * entropy +
                      lambda_req * req_loss +
                      lambda_consistency * consistency_loss)

        logs = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "req_loss": req_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "total_loss": total_loss.item()
        }

        return total_loss, logs

    def compute_a2c_losses(self, states, actions, targets, advantages, shielded_probs,
                           ent_coef=0.01, lambda_req=0.00, lambda_consistency=0.00):
        logits, predicted_values = self.forward(states)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        policy_loss = -(advantages.detach() * new_log_probs).mean()
        value_loss = nn.MSELoss()(predicted_values.squeeze(), targets)

        goal = torch.zeros_like(shielded_probs)
        goal.scatter_(1, actions.unsqueeze(1), 1.0)
        req_loss = nn.BCELoss()(shielded_probs, goal)
        consistency_loss = nn.MSELoss()(torch.softmax(logits, dim=-1), shielded_probs)

        loss = policy_loss + value_loss + lambda_req * req_loss + lambda_consistency * consistency_loss - ent_coef * entropy

        logs = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "req_loss": req_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "total_loss": loss.item()
        }

        return loss, logs

    def compute_q_loss(self, states, actions, rewards, dones, next_states, target_network,
                       gamma=0.99, shielded_probs=None, lambda_req=0.00, lambda_consistency=0.00):
        q_out = self.forward(states)
        q_values = q_out.gather(1, actions)

        with torch.no_grad():
            next_q = target_network(next_states)
            next_max_q = next_q.max(1, keepdim=True)[0]
            target_q_values = rewards + gamma * next_max_q * (1 - dones)

        td_loss = nn.MSELoss()(q_values, target_q_values)

        raw_probs = torch.softmax(q_out, dim=1)

        if shielded_probs is None:
            shielded_probs = raw_probs.detach()

        goal = torch.zeros_like(shielded_probs)
        goal.scatter_(1, actions, 1.0)

        req_loss = nn.BCELoss()(shielded_probs, goal)
        consistency_loss = nn.MSELoss()(raw_probs, shielded_probs)

        total_loss = td_loss + lambda_req * req_loss + lambda_consistency * consistency_loss
        logs = {
            "td_loss": td_loss.item(),
            "req_loss": req_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "total_loss": total_loss.item(),
            "prob_shift": torch.abs(shielded_probs - raw_probs).mean().item()
        }

        return total_loss, logs
