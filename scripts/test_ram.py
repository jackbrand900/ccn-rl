import gymnasium as gym
import ale_py           # ensures Atari envs—including RAM variants—are registered
import numpy as np
from gymnasium.envs.registration import registry

freeway_envs = [env_id for env_id in registry.keys() if "Freeway" in env_id]
print(freeway_envs)

env = gym.make("ALE/Freeway-v5", frameskip=1)
obs, _ = env.reset()

for step in range(100):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    ram = env.unwrapped.ale.getRAM()
    print(f"Step {step}, RAM sum: {ram.sum()}, RAM[0:10]: {ram[:10]}")

    if terminated or truncated:
        print("Episode finished; resetting…")
        obs, _ = env.reset()

env.close()
