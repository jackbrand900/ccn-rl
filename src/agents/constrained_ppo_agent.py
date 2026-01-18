import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from src.models.network import ModularNetwork
from src.utils.preprocessing import prepare_input, prepare_batch
from src.utils.context_provider import build_context
from src.utils.constraint_monitor import ConstraintMonitor
from src.utils.shield_controller import ShieldController


class ConstrainedPPOAgent:
    def __init__(self, input_shape, action_dim,
                 agent_kwargs=None,
                 use_cnn=False,
                 env=None,
                 requirements_path=None,
                 monitor_constraints=True,
                 mode='hard',
                 budget=None,  # Will be set from agent_kwargs if provided
                 nu_lr=1e-3,
                 verbose=False):

        self.env = env
        self.use_cnn = use_cnn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.memory = []
        self.monitor_constraints = monitor_constraints
        self.use_shield_pre = False
        self.use_shield_post = False
        self.use_shield_layer = False

        # === Hyperparameters ===
        agent_kwargs = agent_kwargs or {}
        # Default hyperparameters (can be overridden)
        default_kwargs = {'lr': 0.0017, 'gamma': 0.97, 'hidden_dim': 512, 'use_orthogonal_init': True,
                         'num_layers': 4, 'cost_gamma': 0.96, 'cost_lam': 0.94, 'clip_eps': 0.18,
                         'ent_coef': 0.018, 'epochs': 10, 'batch_size': 64, 'budget': 0.15, 'nu_lr': 1e-3,
                         'max_episode_memory': 5000}  # Limit memory per episode to prevent OOM
        # Merge with provided kwargs
        for key, val in default_kwargs.items():
            if key not in agent_kwargs:
                agent_kwargs[key] = val
        
        # Budget can be set via agent_kwargs or constructor parameter
        if budget is None:
            budget = agent_kwargs.get('budget', 0.15)
        if 'nu_lr' in agent_kwargs:
            nu_lr = agent_kwargs['nu_lr']
        
        self.max_episode_memory = agent_kwargs.get('max_episode_memory', 5000)
        self.hidden_dim = agent_kwargs.get("hidden_dim", 128)
        self.use_orthogonal_init = agent_kwargs.get("use_orthogonal_init", False)
        self.lr = agent_kwargs.get("lr", 3e-4)
        self.gamma = agent_kwargs.get("gamma", 0.99)
        self.cost_gamma = agent_kwargs.get("cost_gamma", 0.99)
        self.lam = agent_kwargs.get("lam", 0.95)
        self.cost_lam = agent_kwargs.get("cost_lam", 0.95)
        self.clip_eps = agent_kwargs.get("clip_eps", 0.2)
        self.ent_coef = agent_kwargs.get("ent_coef", 0.01)
        self.batch_size = agent_kwargs.get("batch_size", 64)
        self.epochs = agent_kwargs.get("epochs", 10)
        self.num_layers = agent_kwargs.get("num_layers", 3)
        self.lambda_sem = 0
        self.learn_step_counter = 0
        self.verbose = verbose  # Store verbose flag

        # === Networks ===
        self.policy_net = ModularNetwork(input_shape, action_dim, self.hidden_dim,
                                         self.num_layers, self.use_cnn, None,
                                         self.use_orthogonal_init).to(self.device)

        self.value_net = ModularNetwork(input_shape, 1, self.hidden_dim,
                                        self.num_layers, self.use_cnn, None,
                                        self.use_orthogonal_init).to(self.device)

        self.cost_value_net = ModularNetwork(input_shape, 1, self.hidden_dim,
                                             self.num_layers, self.use_cnn, None,
                                             self.use_orthogonal_init).to(self.device)

        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=self.lr)
        self.cost_optimizer = optim.Adam(self.cost_value_net.parameters(), lr=self.lr)

        # === Constraints ===
        self.constraint_monitor = ConstraintMonitor(verbose=verbose)
        self.shield_controller = ShieldController(requirements_path, action_dim, mode,
                                                  verbose=verbose, is_shield_active=False)
        self.shield_controller.constraint_monitor = self.constraint_monitor

        # === Lagrangian multiplier ===
        self.budget = budget
        self.nu = torch.tensor(1.0, requires_grad=True, device=self.device)
        self.nu_lr = nu_lr
        self.nu_optim = optim.Adam([self.nu], lr=nu_lr)

    def compute_gae(self, rewards, values, dones, gamma, lam):
        advantages, gae = [], 0
        values = values + [0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return returns, advantages

    def select_action(self, state, env=None, do_apply_shield=True):
        self.last_obs = state
        context = build_context(env or self.env, self)

        state_tensor = prepare_input(state, use_cnn=self.use_cnn).to(self.device)
        # print("[DEBUG] state_tensor shape before CNN:", state_tensor.shape)

        logits = self.policy_net(state_tensor)
        if isinstance(logits, tuple):  # (logits, features)
            logits = logits[0]

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value_net(state_tensor)
        if isinstance(value, tuple):
            value = value[0]
        value = value.squeeze()
        raw_probs = torch.softmax(logits, dim=-1).squeeze(0)
        shielded_probs = self.shield_controller.apply(raw_probs.unsqueeze(0), context).squeeze(0)
        shielded_probs /= shielded_probs.sum()

        dist_shielded = Categorical(probs=shielded_probs)
        a_shielded = dist_shielded.sample().item()
        a_unshielded = action.item()

        self.constraint_monitor.log_step_from_probs_and_actions(
            raw_probs=raw_probs.detach(),
            corrected_probs=shielded_probs.detach(),
            a_unshielded=a_unshielded,
            a_shielded=a_shielded,
            context=context,
            shield_controller=self.shield_controller,
        )

        # Compute cost: 1.0 if action would violate constraints, 0.0 otherwise
        # Cost is based on whether the unshielded action would violate
        cost = 0.0
        if self.shield_controller and context:
            # Check if flags are active
            flags = self.shield_controller.flag_logic_fn(context)
            active_flags = {name: val for name, val in flags.items()
                          if name in self.shield_controller.flag_names and val > 0}
            
            if len(active_flags) > 0:
                # Flags are active, check if unshielded action would violate
                would_violate = self.constraint_monitor.would_violate(
                    a_unshielded, context, self.shield_controller
                )
                cost = 1.0 if would_violate else 0.0

        # For memory efficiency, manage memory size for very long episodes
        # If memory is getting too large, remove oldest entries
        if len(self.memory) >= self.max_episode_memory:
            # Remove oldest 10% to make room
            remove_count = self.max_episode_memory // 10
            self.memory = self.memory[remove_count:]
            if self.verbose and len(self.memory) % 1000 == 0:
                print(f"[CPPO] Memory management: removed {remove_count} old transitions")

        self.memory.append({
            'state': state,
            'action': action.item(),  # Store unshielded action for learning
            'reward': 0.0,
            'next_state': None,
            'done': False,
            'log_prob': log_prob.item(),
            'value': value.item(),
            'cost': cost,
            'context': context
        })

        # Return shielded action for execution (safe), but we learn from unshielded
        return a_shielded, a_unshielded, a_shielded, context

    def store_transition(self, state, action, reward, next_state, context, done):
        if not self.memory:
            raise ValueError("No memory to update. Call select_action first.")
        self.memory[-1]['reward'] = reward
        self.memory[-1]['next_state'] = next_state
        self.memory[-1]['done'] = done

    def update(self):
        # For very long episodes, update periodically to prevent memory issues
        # Check if we should update (either episode done OR memory limit reached)
        should_update = False
        
        if self.memory:
            # Update if episode is done
            if self.memory[-1].get('done', False):
                should_update = True
            # Or if memory limit reached (for very long episodes)
            elif len(self.memory) >= self.max_episode_memory:
                should_update = True
                if self.verbose:
                    print(f"[CPPO] Memory limit reached ({len(self.memory)} transitions), updating mid-episode")
        
        if not should_update:
            return
        
        # Need at least batch_size transitions to update
        if len(self.memory) < self.batch_size:
            return
        
        # For very long episodes, limit memory to prevent OOM
        if len(self.memory) > self.max_episode_memory:
            # Keep the most recent transitions
            self.memory = self.memory[-self.max_episode_memory:]
            if self.verbose:
                print(f"[CPPO] Memory truncated to {self.max_episode_memory} transitions to prevent OOM")
        
        self.learn_step_counter += 1
        
        # Extract all transitions from memory (full rollout)
        states = prepare_batch([t['state'] for t in self.memory], use_cnn=self.use_cnn).to(self.device)
        actions = torch.tensor([t['action'] for t in self.memory]).to(self.device)
        rewards = [t['reward'] for t in self.memory]
        costs = [t['cost'] for t in self.memory]
        dones = [t['done'] for t in self.memory]
        old_log_probs = torch.tensor([t['log_prob'] for t in self.memory]).to(self.device)
        values = [t['value'] for t in self.memory]

        # Compute cost values for all states
        with torch.no_grad():
            cost_values = []
            for s in states:
                out = self.cost_value_net(s.unsqueeze(0))
                value = out[0] if isinstance(out, tuple) else out
                cost_values.append(value.item())

        # Compute GAE for full rollout
        returns, adv = self.compute_gae(rewards, values, dones, self.gamma, self.lam)
        cost_returns, cost_adv = self.compute_gae(costs, cost_values, dones, self.cost_gamma, self.cost_lam)

        adv = torch.tensor(adv).float().to(self.device)
        cost_adv = torch.tensor(cost_adv).float().to(self.device)
        returns = torch.tensor(returns).float().to(self.device)
        cost_returns = torch.tensor(cost_returns).float().to(self.device)
        
        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        cost_adv = (cost_adv - cost_adv.mean()) / (cost_adv.std() + 1e-8)

        # Multiple epochs over the rollout
        for epoch in range(self.epochs):
            # Shuffle for each epoch
            indices = torch.randperm(len(self.memory)).to(self.device)
            
            # Mini-batch updates
            for start in range(0, len(self.memory), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_adv = adv[batch_indices]
                batch_cost_adv = cost_adv[batch_indices]
                batch_returns = returns[batch_indices]
                batch_cost_returns = cost_returns[batch_indices]

                # === PPO Policy update ===
                logits = self.policy_net(batch_states)
                if isinstance(logits, tuple):  # (logits, features)
                    logits = logits[0]

                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Cost loss: we want to minimize cost, so use negative cost advantage
                # cost_adv is positive when we want to reduce cost (high cost)
                cost_surr1 = ratio * batch_cost_adv
                cost_surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_cost_adv
                cost_loss = torch.min(cost_surr1, cost_surr2).mean()
                
                # Total loss: maximize reward, minimize cost (subject to budget)
                # nu * cost_loss penalizes high costs when nu > 0
                nu_value = self.nu.clamp(min=0.0, max=10.0)  # Clamp nu for stability
                total_loss = policy_loss + nu_value * cost_loss - self.ent_coef * entropy
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.policy_net.parameters()) + list(self.value_net.parameters()), max_norm=0.5)
                self.optimizer.step()

                # Value network update
                value_out = self.value_net(batch_states)
                if isinstance(value_out, tuple):
                    value_out = value_out[0]
                value_preds = value_out.squeeze(-1)  # Only squeeze last dimension to preserve batch dimension
                # Ensure shapes match (handle edge case of batch_size=1)
                if value_preds.dim() == 0:
                    value_preds = value_preds.unsqueeze(0)
                if batch_returns.dim() == 0:
                    batch_returns = batch_returns.unsqueeze(0)
                value_loss = nn.MSELoss()(value_preds, batch_returns)
                self.optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.policy_net.parameters()) + list(self.value_net.parameters()), max_norm=0.5)
                self.optimizer.step()

                # Cost value network update
                cost_out = self.cost_value_net(batch_states)
                if isinstance(cost_out, tuple):
                    cost_out = cost_out[0]
                cost_value_preds = cost_out.squeeze(-1)  # Only squeeze last dimension to preserve batch dimension
                # Ensure shapes match (handle edge case of batch_size=1)
                if cost_value_preds.dim() == 0:
                    cost_value_preds = cost_value_preds.unsqueeze(0)
                if batch_cost_returns.dim() == 0:
                    batch_cost_returns = batch_cost_returns.unsqueeze(0)
                cost_value_loss = nn.MSELoss()(cost_value_preds, batch_cost_returns)
                self.cost_optimizer.zero_grad()
                cost_value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cost_value_net.parameters(), max_norm=0.5)
                self.cost_optimizer.step()

        # Update Lagrangian multiplier (once per episode)
        avg_cost = torch.tensor(costs).float().mean().to(self.device)
        # Clamp nu to prevent it from growing too large too quickly
        # Update: nu = max(0, nu + lr * (avg_cost - budget))
        cost_violation = (avg_cost - self.budget).item()
        old_nu = self.nu.item()
        
        if cost_violation > 0:
            # Increase nu if we're violating the budget
            self.nu.data = torch.clamp(self.nu.data + self.nu_lr * cost_violation, min=0.0, max=10.0)  # Reduced max from 100 to 10
        else:
            # Decrease nu if we're under budget (but don't go negative)
            # Decay more slowly when under budget to avoid oscillations
            decay_rate = 0.5  # Decay by 50% when under budget
            self.nu.data = torch.clamp(self.nu.data * (1 - self.nu_lr * decay_rate), min=0.0)
        
        # Debug logging (every 50 episodes)
        if self.verbose and self.learn_step_counter % 50 == 0:
            print(f"[CPPO Debug] Episode {self.learn_step_counter}, Avg Cost: {avg_cost.item():.4f}, "
                  f"Budget: {self.budget:.4f}, Nu: {old_nu:.4f} -> {self.nu.item():.4f}, "
                  f"Cost Violation: {cost_violation:.4f}")

        # Clear memory after update (ready for next episode)
        self.memory.clear()
        
        # Explicit garbage collection for large environments
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_weights(self):
        return {
            "policy": self.policy_net.state_dict(),
            "value": self.value_net.state_dict(),
            "cost": self.cost_value_net.state_dict()
        }

    def load_weights(self, weights):
        self.policy_net.load_state_dict(weights["policy"])
        self.value_net.load_state_dict(weights["value"])
        self.cost_value_net.load_state_dict(weights["cost"])
