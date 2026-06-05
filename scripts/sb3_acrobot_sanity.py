"""SB3-default PPO sanity check on Acrobot, 10 seeds, two obs conditions:
  raw  : gym.make('Acrobot-v1')  (matches the current wrapper-removed env)
  norm : + VecNormalize(norm_obs)  (the standard fix for the velocity scale gap
         that the removed AcrobotNormalizeObs wrapper used to handle)

Decisive question: is Acrobot's 7/10 vanilla flooring an env/obs-scale problem
or purely a custom-agent tuning artifact?
  - raw solves 10/10  -> tuning artifact (the custom config is broken)
  - raw floors, norm solves -> removing the obs wrapper caused it; restore norm
"""
import warnings; warnings.filterwarnings("ignore")
import statistics as st
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

SEEDS = [42, 123, 456, 789, 1011, 2024, 1337, 7, 314, 271]
TIMESTEPS = 150_000

def run(seed, normalize):
    venv = DummyVecEnv([lambda: gym.make("Acrobot-v1")])
    if normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False)
    model = PPO("MlpPolicy", venv, seed=seed, verbose=0)
    model.learn(total_timesteps=TIMESTEPS)
    # eval env
    eval_env = DummyVecEnv([lambda: gym.make("Acrobot-v1")])
    if normalize:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
        eval_env.obs_rms = venv.obs_rms  # reuse training obs stats
    mean_r, _ = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    return mean_r

for cond, normalize in [("raw", False), ("norm", True)]:
    rewards = []
    for seed in SEEDS:
        r = run(seed, normalize)
        rewards.append(r)
        print(f"[SB3 {cond} seed={seed}] reward={r:.1f}", flush=True)
    med = st.median(rewards)
    solved = sum(1 for x in rewards if x >= -130)
    floored = sum(1 for x in rewards if x <= -450)
    print(f"=== SB3 {cond}: mean={st.mean(rewards):.1f} median={med:.1f} "
          f"solved>=-130 {solved}/10  floored {floored}/10 ===", flush=True)
