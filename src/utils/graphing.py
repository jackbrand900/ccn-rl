import matplotlib.pyplot as plt
import numpy as np
import os

def plot_rewards(
        rewards,
        title="Training Rewards",
        xlabel="Episode",
        ylabel="Reward",
        rolling_window=10,
        save_path=None,
        show=True
):
    """
    Plot rewards over episodes with optional rolling average.

    Args:
        rewards (list or np.array): List of rewards per episode.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        rolling_window (int): Window size for smoothing.
        save_path (str or None): If provided, save the plot to this path.
        show (bool): If True, display the plot.
    """
    episodes = np.arange(1, len(rewards) + 1)
    rewards = np.array(rewards)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, label="Episode Reward", alpha=0.6)

    if rolling_window > 1:
        rolling_mean = np.convolve(
            rewards, np.ones(rolling_window) / rolling_window, mode='valid'
        )
        rolling_episodes = np.arange(rolling_window, len(rewards) + 1)
        plt.plot(rolling_episodes, rolling_mean, label=f"{rolling_window}-Episode Average", linewidth=2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
