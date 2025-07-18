import matplotlib.pyplot as plt
import numpy as np
import os
import math
from collections import defaultdict

def plot_rewards(
        rewards,
        title="Training Rewards",
        xlabel="Episode",
        ylabel="Reward",
        rolling_window=10,
        save_path=None,
        show=True
):
    episodes = np.arange(1, len(rewards) + 1)
    rewards = np.array(rewards)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Plot raw rewards
    ax.plot(episodes, rewards, label="Episode Reward", alpha=0.6)

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    # Rolling mean and std
    if rolling_window > 1:
        rolling_mean = np.convolve(rewards, np.ones(rolling_window) / rolling_window, mode='valid')
        rolling_std = np.std([rewards[max(0, i - rolling_window):i] for i in range(rolling_window, len(rewards) + 1)], axis=1)
        rolling_episodes = np.arange(rolling_window, len(rewards) + 1)

        ax.plot(rolling_episodes, rolling_mean, label=f"{rolling_window}-Episode Average", linewidth=2)
        ax.fill_between(rolling_episodes,
                        rolling_mean - rolling_std,
                        rolling_mean + rolling_std,
                        alpha=0.2,
                        color="gray",
                        label="±1 Std Dev")

    # Add stats to top-left
    stats_text = f"Mean: {avg_reward:.2f}\nStd Dev: {std_reward:.2f}"
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))

    # Add legend to top-right
    ax.legend(loc='upper right')

    # Labels and formatting
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_action_frequencies(actions, action_labels=None, title="Action Selection Frequencies", save_path=None, show=True):
    """
    Plot a histogram showing the frequency of each action selected.

    Args:
        actions (list or np.array): List of actions selected (integers).
        action_labels (list or None): List of labels for actions, e.g. ['Left', 'Right', ...]. If None, uses indices.
        title (str): Title of the plot.
        save_path (str or None): If provided, saves the plot to this path.
        show (bool): If True, displays the plot.
    """
    actions = np.array(actions)
    unique_actions, counts = np.unique(actions, return_counts=True)

    plt.figure(figsize=(10, 6))
    plt.bar(unique_actions, counts, color='skyblue', edgecolor='black')

    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.title(title)

    if action_labels:
        plt.xticks(unique_actions, [action_labels[i] for i in unique_actions])
    else:
        plt.xticks(unique_actions, unique_actions)

    plt.grid(axis='y', alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

def plot_action_heatmaps(visited_actions, grid_size=(5, 5), num_actions=7):
    """
    Visualizes how often each action was taken in each grid cell.
    Shows one heatmap per action (not just the most frequent).

    Args:
        visited_actions (list of (x, y, action)): List of agent steps (1-indexed).
        grid_size (tuple): (width, height) of the environment.
        num_actions (int): Number of discrete actions.
    """
    w, h = grid_size
    action_names = [
        "left", "right", "forward", "pickup", "drop", "toggle", "done"
    ]

    # Initialize grid for each action
    action_grids = np.zeros((num_actions, h, w))  # shape: (action, row, col)

    for x, y, action in visited_actions:
        xi, yi = x - 1, y - 1  # Convert to 0-indexed
        if 0 <= xi < w and 0 <= yi < h and 0 <= action < num_actions:
            action_grids[action, h - yi - 1, xi] += 1  # Flip y for display

    # Plot one heatmap per action
    fig, axs = plt.subplots(1, num_actions, figsize=(4 * num_actions, 4))

    for i in range(num_actions):
        ax = axs[i]
        im = ax.imshow(action_grids[i], cmap='YlOrRd', origin='upper')
        ax.set_title(f"Action: {action_names[i]}")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(w))
        ax.set_yticks(np.arange(h))
        ax.set_xticklabels([str(i + 1) for i in range(w)])
        ax.set_yticklabels([str(h - i) for i in range(h)])
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def get_action_counts_per_state(visited_actions, grid_size=(5, 5), num_actions=7):
    """
    Print the frequency of each action taken at every state (1-indexed).

    Args:
        visited_actions (list of (x, y, action)): List of visited states and actions.
        grid_size (tuple): Size of the grid (width, height).
        num_actions (int): Number of possible actions.
    """
    action_names = [
        "left", "right", "forward", "pickup", "drop", "toggle", "done"
    ]
    action_counts = defaultdict(lambda: np.zeros(num_actions))

    for x, y, action in visited_actions:
        xi, yi = x - 1, y - 1
        if 0 <= xi < grid_size[0] and 0 <= yi < grid_size[1] and 0 <= action < num_actions:
            action_counts[(x, y)][action] += 1

    print("Action Frequencies per State:")
    for (x, y), counts in sorted(action_counts.items()):
        counts_str = ", ".join(f"{action_names[i]}: {int(c)}" for i, c in enumerate(counts))
        print(f"State ({x}, {y}): {counts_str}")

    return action_counts

def plot_action_histograms(action_counts, action_names=None, max_cols=10):
    if action_names is None:
        action_names = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]

    states = sorted(action_counts.keys())
    num_states = len(states)
    num_cols = min(num_states, max_cols)
    num_rows = math.ceil(num_states / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows), squeeze=False)
    for idx, (state, counts) in enumerate(sorted(action_counts.items())):
        row, col = divmod(idx, num_cols)
        ax = axes[row][col]
        ax.bar(action_names, counts)
        ax.set_title(f"State {state}")
        ax.set_xticklabels(action_names, rotation=45)
        ax.set_ylabel("Freq")

    # Hide unused subplots
    for idx in range(num_states, num_rows * num_cols):
        row, col = divmod(idx, num_cols)
        axes[row][col].axis('off')

    plt.tight_layout()
    plt.show()

def moving_average(values, window=10):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode='valid')

def plot_losses(logs, window=10, save_path=None):
    keys = ["td_loss", "req_loss", "consistency_loss", "policy_loss", "value_loss"]
    available = [k for k in keys if k in logs]
    smoothed = {k: moving_average(logs[k], window) for k in available}

    if not smoothed:
        print("No loss keys found in logs.")
        return

    # Subplots: one per available loss
    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5), sharex=True)

    if len(available) == 1:
        axes = [axes]

    for ax, key in zip(axes, available):
        ax.plot(smoothed[key])
        ax.set_title(key.replace("_", " ").title())
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Loss")
        ax.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_prob_shift(logs, window=10, save_path=None):
    prob_shift = moving_average(logs["prob_shift"], window)
    smoothed_steps = range(len(prob_shift))

    plt.figure(figsize=(8, 5))
    plt.plot(smoothed_steps, prob_shift, label="Shield Probability Shift", color="orange")
    plt.title("Shield-Induced Probability Shift (Smoothed)")
    plt.xlabel("Training Steps")
    plt.ylabel("Avg L1 Shift in Action Probabilities")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_violations(
        violations,
        total_steps,
        title="Constraint Violations per Episode",
        save_path=None,
        show=True
):
    episodes = np.arange(1, len(violations) + 1)
    violations = np.array(violations)

    total_violations = int(np.sum(violations))
    avg_violations_per_step = total_violations / total_steps if total_steps > 0 else 0

    plt.figure(figsize=(10, 4))
    ax = plt.gca()

    ax.plot(episodes, violations, color='red', label="Violations")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Violations")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="upper right")

    # Top-left annotation
    stats_text = (
        f"Total Violations: {total_violations}\n"
        f"Avg/Step: {avg_violations_per_step:.4f}"
    )
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Violation plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


