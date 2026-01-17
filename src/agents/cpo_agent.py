"""
Constrained Policy Optimization (CPO) Agent
Based on: Achiam et al. "Constrained Policy Optimization" (ICML 2017)
"""

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


class CPOAgent:
    def __init__(self, input_shape, action_dim,
                 agent_kwargs=None,
                 use_cnn=False,
                 env=None,
                 requirements_path=None,
                 monitor_constraints=True,
                 mode='hard',
                 budget=0.05,
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
        self.verbose = verbose

        # === Hyperparameters ===
        agent_kwargs = agent_kwargs or {}
        default_kwargs = {
            'lr': 3e-4, 'gamma': 0.99, 'hidden_dim': 128, 'use_orthogonal_init': False,
            'num_layers': 2, 'cost_gamma': 0.99, 'cost_lam': 0.95, 'clip_eps': 0.2,
            'ent_coef': 0.01, 'epochs': 10, 'batch_size': 64, 'budget': 0.05,
            'max_kl': 0.01, 'damping': 0.1, 'max_backtrack': 10, 'backtrack_coef': 0.8,
            'max_episode_memory': 5000
        }
        for key, val in default_kwargs.items():
            if key not in agent_kwargs:
                agent_kwargs[key] = val
        
        if budget is None:
            budget = agent_kwargs.get('budget', 0.05)
        
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
        self.num_layers = agent_kwargs.get("num_layers", 2)
        self.max_kl = agent_kwargs.get("max_kl", 0.01)  # Trust region size
        self.damping = agent_kwargs.get("damping", 0.1)  # For numerical stability
        self.max_backtrack = agent_kwargs.get("max_backtrack", 10)
        self.backtrack_coef = agent_kwargs.get("backtrack_coef", 0.8)
        self.max_episode_memory = agent_kwargs.get("max_episode_memory", 5000)
        self.lambda_sem = 0
        self.learn_step_counter = 0

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

        # === CPO parameters ===
        self.budget = budget

    def compute_gae(self, rewards, values, dones, gamma, lam):
        advantages, gae = [], 0
        values = values + [0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    def select_action(self, state, env=None, do_apply_shield=True):
        self.last_obs = state
        context = build_context(env or self.env, self)

        state_tensor = prepare_input(state, use_cnn=self.use_cnn).to(self.device)
        logits = self.policy_net(state_tensor)
        if isinstance(logits, tuple):
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
        cost = 0.0
        if self.shield_controller and context:
            flags = self.shield_controller.flag_logic_fn(context)
            active_flags = {name: val for name, val in flags.items()
                          if name in self.shield_controller.flag_names and val > 0}

            if len(active_flags) > 0:
                would_violate = self.constraint_monitor.would_violate(
                    a_unshielded, context, self.shield_controller
                )
                cost = 1.0 if would_violate else 0.0

        # Memory management
        if len(self.memory) >= self.max_episode_memory:
            remove_count = self.max_episode_memory // 10
            self.memory = self.memory[remove_count:]
            if self.verbose and len(self.memory) % 1000 == 0:
                print(f"[CPO] Memory management: removed {remove_count} old transitions")

        self.memory.append({
            'state': state,
            'action': action.item(),
            'reward': 0.0,
            'next_state': None,
            'done': False,
            'log_prob': log_prob.item(),
            'value': value.item(),
            'cost': cost,
            'context': context
        })

        return a_shielded, a_unshielded, a_shielded, context

    def store_transition(self, state, action, reward, next_state, context, done):
        if not self.memory:
            raise ValueError("No memory to update. Call select_action first.")
        self.memory[-1]['reward'] = reward
        self.memory[-1]['next_state'] = next_state
        self.memory[-1]['done'] = done

    def _flat_grad(self, loss, params, create_graph=False, retain_graph=False):
        """Compute flat gradient"""
        grads = torch.autograd.grad(loss, params, create_graph=create_graph, retain_graph=retain_graph)
        return torch.cat([g.view(-1) for g in grads])

    def _conjugate_gradient(self, Ax, b, n_iter=10, residual_tol=1e-10):
        """Conjugate gradient algorithm for solving Ax = b"""
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        r_dot_old = torch.dot(r, r)
        
        for _ in range(n_iter):
            Ap = Ax(p)
            alpha = r_dot_old / (torch.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            r_dot_new = torch.dot(r, r)
            if r_dot_new < residual_tol:
                break
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        
        return x

    def _fisher_vector_product(self, kl, params):
        """Compute Fisher-vector product for natural gradient"""
        grads = torch.autograd.grad(kl, params, create_graph=True)
        flat_grads = torch.cat([g.view(-1) for g in grads])
        
        def fisher_vec_prod(v):
            grad_grad = torch.autograd.grad(flat_grads, params, v, retain_graph=True)
            return torch.cat([g.contiguous().view(-1) for g in grad_grad]) + self.damping * v
        
        return fisher_vec_prod

    def update(self):
        # Only update at end of episode
        if not self.memory or not self.memory[-1].get('done', False):
            return
        
        if len(self.memory) < self.batch_size:
            return
        
        if len(self.memory) > self.max_episode_memory:
            self.memory = self.memory[-self.max_episode_memory:]
            if self.verbose:
                print(f"[CPO] Memory truncated to {self.max_episode_memory} transitions")

        self.learn_step_counter += 1

        # Extract transitions
        states = prepare_batch([t['state'] for t in self.memory], use_cnn=self.use_cnn).to(self.device)
        actions = torch.tensor([t['action'] for t in self.memory]).to(self.device)
        rewards = [t['reward'] for t in self.memory]
        costs = [t['cost'] for t in self.memory]
        dones = [t['done'] for t in self.memory]
        old_log_probs = torch.tensor([t['log_prob'] for t in self.memory]).to(self.device)
        values = [t['value'] for t in self.memory]

        # Compute GAE for rewards and costs
        with torch.no_grad():
            next_values = []
            for i, s in enumerate(states):
                v = self.value_net(s.unsqueeze(0))
                if isinstance(v, tuple):
                    v = v[0]
                next_values.append(v.squeeze().item())
            next_values.append(0.0)
            
            adv = self.compute_gae(rewards, values + [0], dones, self.gamma, self.lam)
            cost_adv = self.compute_gae(costs, [0] * len(costs) + [0], dones, self.cost_gamma, self.cost_lam)
            
            returns = [adv[i] + values[i] for i in range(len(adv))]
            cost_returns = [cost_adv[i] for i in range(len(cost_adv))]

        adv = torch.tensor(adv).float().to(self.device)
        cost_adv = torch.tensor(cost_adv).float().to(self.device)
        returns = torch.tensor(returns).float().to(self.device)
        cost_returns = torch.tensor(cost_returns).float().to(self.device)

        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        cost_adv = (cost_adv - cost_adv.mean()) / (cost_adv.std() + 1e-8)

        # Get current policy parameters
        policy_params = list(self.policy_net.parameters())

        # Compute policy gradient (reward)
        logits = self.policy_net(states)
        if isinstance(logits, tuple):
            logits = logits[0]
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        policy_loss = -(new_log_probs * adv).mean()
        
        # Compute cost gradient
        cost_loss = (new_log_probs * cost_adv).mean()
        
        # Compute KL divergence for trust region
        kl = (old_log_probs - new_log_probs).mean()
        
        # CPO update: solve constrained optimization problem
        # max g^T * s subject to: ||s||_H <= delta and a^T * s <= c
        # where g = reward gradient, a = cost gradient, H = Fisher info matrix
        
        # Get flat gradients
        g = self._flat_grad(policy_loss, policy_params, retain_graph=True)
        a = self._flat_grad(cost_loss, policy_params, retain_graph=True)
        
        # Compute average cost
        avg_cost = torch.tensor(costs).float().mean()
        cost_violation = (avg_cost - self.budget).item()
        
        # If constraint is satisfied, do standard policy gradient
        if cost_violation <= 0:
            # Unconstrained update
            step_direction = g
        else:
            # Constrained update: need to project onto constraint set
            # For simplicity, use Lagrangian with adaptive step size
            # Full CPO would solve the quadratic program, but this is a simplified version
            
            # Compute Fisher-vector product function
            kl_loss = kl
            fisher_vec_prod = self._fisher_vector_product(kl_loss, policy_params)
            
            # Solve for natural gradient direction
            try:
                # Use conjugate gradient to solve Hx = g
                x = self._conjugate_gradient(fisher_vec_prod, g, n_iter=10)
                
                # Project onto constraint: if a^T * x > c, scale down
                a_dot_x = torch.dot(a, x)
                if a_dot_x > cost_violation:
                    # Scale to satisfy constraint
                    scale = cost_violation / (a_dot_x + 1e-8)
                    x = scale * x
                
                # Scale to satisfy trust region
                x_Hx = torch.dot(x, fisher_vec_prod(x))
                if x_Hx > self.max_kl:
                    scale = torch.sqrt(self.max_kl / (x_Hx + 1e-8))
                    x = scale * x
                
                step_direction = x
            except:
                # Fallback: simple gradient with constraint penalty
                step_direction = g - 0.1 * a
        
        # Update policy using computed direction
        # Convert flat gradient back to parameter updates
        idx = 0
        for param in policy_params:
            param_size = param.numel()
            param_update = step_direction[idx:idx + param_size].view(param.shape)
            param.data.add_(param_update, alpha=self.lr)
            idx += param_size

        # Update value networks
        value_out = self.value_net(states)
        if isinstance(value_out, tuple):
            value_out = value_out[0]
        value_preds = value_out.squeeze()
        value_loss = nn.MSELoss()(value_preds, returns)
        self.optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.policy_net.parameters()) + list(self.value_net.parameters()), max_norm=0.5)
        self.optimizer.step()

        cost_out = self.cost_value_net(states)
        if isinstance(cost_out, tuple):
            cost_out = cost_out[0]
        cost_value_preds = cost_out.squeeze()
        cost_value_loss = nn.MSELoss()(cost_value_preds, cost_returns)
        self.cost_optimizer.zero_grad()
        cost_value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cost_value_net.parameters(), max_norm=0.5)
        self.cost_optimizer.step()

        # Clear memory
        self.memory.clear()
        
        # Garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_weights(self):
        return {
            'policy': self.policy_net.state_dict(),
            'value': self.value_net.state_dict(),
            'cost_value': self.cost_value_net.state_dict()
        }

    def load_weights(self, weights):
        self.policy_net.load_state_dict(weights['policy'])
        self.value_net.load_state_dict(weights['value'])
        self.cost_value_net.load_state_dict(weights['cost_value'])

