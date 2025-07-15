import os
from itertools import product

import gymnasium as gym
from src.agents.ppo_agent import PPOAgent
from src.train import create_environment, run_training
from src.utils.context_provider import cartpole_flag_logic_advanced

# Sweep ranges
theta_thresh_vals = [0.01, 0.05, 0.1]
pos_thresh_vals = [0.5, 0.8, 1.0]
emergency_thresh_vals = [0.1, 0.15, 0.2]

# Where to save logs
results_dir = "flag_sweep_results"
os.makedirs(results_dir, exist_ok=True)

# Run all combinations
for theta_thresh, pos_thresh, emergency_thresh in product(theta_thresh_vals, pos_thresh_vals, emergency_thresh_vals):
    name = f"theta{theta_thresh}_pos{pos_thresh}_emg{emergency_thresh}"
    print(f"\n--- Running {name} ---")

    env_name = "CartPole-v1"
    env = create_environment(env_name)

    obs_space = env.observation_space
    state_dim = int(obs_space.shape[0])
    action_dim = env.action_space.n

    # Construct shielded agent with custom flag logic
    def custom_flag_logic(ctx):
        return cartpole_flag_logic_advanced(
            ctx,
            theta_thresh=theta_thresh,
            pos_thresh=pos_thresh,
            emergency_thresh=emergency_thresh,
        )

    agent = PPOAgent(
        state_dim,
        action_dim,
        use_shield=True,
        requirements_path="src/requirements/advanced_cartpole.linear",
        env=env,
        verbose=False,
    )

    agent.shield_controller.flag_logic_fn = custom_flag_logic

    rewards = run_training(agent, env, num_episodes=500, print_interval=20, visualize=False, log_rewards=True)
    avg_reward = sum(rewards[-10:]) / 10
    print(f"[{name}] Avg Reward (last 10): {avg_reward:.2f}")
