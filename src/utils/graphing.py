from datetime import datetime
import matplotlib
import pandas as pd
from setuptools import glob

matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import os
import math
from collections import defaultdict
import matplotlib.patches as mpatches

import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def plot_rewards(
        rewards,
        title="Training Rewards",
        xlabel="Episode",
        ylabel="Reward",
        rolling_window=10,
        save_path=None,
        show=True,
        run_dir=None,
        stds=None
):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from datetime import datetime

    episodes = np.arange(1, len(rewards) + 1)
    rewards = np.array(rewards)

    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Use matplotlib's default blue
    default_blue = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

    # === Mean reward line ===
    ax.plot(episodes, rewards, label="Mean Reward", linewidth=2)

    # === Std deviation band (gray) ===
    if stds is not None:
        stds = np.array(stds)
        ax.fill_between(
            episodes,
            rewards - stds,
            rewards + stds,
            alpha=0.25,
            color="gray",
            label="±1 Std Dev (Across Runs)"
        )

    # === Rolling average (orange) ===
    if rolling_window > 1 and len(rewards) >= rolling_window:
        smoothed = np.convolve(rewards, np.ones(rolling_window) / rolling_window, mode='valid')
        smoothed_episodes = episodes[rolling_window - 1:]
        ax.plot(smoothed_episodes, smoothed, label=f"{rolling_window}-Episode Rolling Avg", color='orange', linewidth=2)

    # === Stats box ===
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    stats_text = f"Mean: {avg_reward:.2f}\nStd Dev: {std_reward:.2f}"
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc='best')

    # === Save plot ===
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[Saved] {save_path}")
    elif run_dir:
        os.makedirs(run_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reward_plot_{timestamp}.png"
        full_path = os.path.join(run_dir, filename)
        plt.savefig(full_path)
        print(f"[Saved] {full_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_action_frequencies(actions, action_labels=None, title="Action Selection Frequencies", save_path=None, show=True):
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
    plt.close()

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
    plt.close()

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
    plt.close()

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
    plt.close()

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
    ax.legend(loc="best")

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

    plt.close()


def plot_summary_metrics(
        rewards,
        mod_rate,
        viol_rate,
        title_prefix,
        save_dir="plots",
        rolling_window=10,
        run_dir=None,
        mod_std=None,
        viol_std=None
):
    episodes = np.arange(len(mod_rate))

    fig, ax = plt.subplots(figsize=(10, 8))
    legend_handles = []

    # === Plot modification rate ===
    mod_line, = ax.plot(episodes, mod_rate, color='orange',
                        label=f"Modification Rate (Mean: {np.mean(mod_rate):.2f})")
    legend_handles.append(mod_line)

    if mod_std is not None and len(mod_std) == len(mod_rate):
        ax.fill_between(
            episodes,
            np.array(mod_rate) - np.array(mod_std),
            np.array(mod_rate) + np.array(mod_std),
            color='orange',
            alpha=0.2
        )
        legend_handles.append(mpatches.Patch(color='orange', alpha=0.2, label='±1 Std Dev (Modifications)'))

    # === Plot violation rate ===
    vio_line, = ax.plot(episodes, viol_rate, color='red',
                        label=f"Violation Rate (Mean: {np.mean(viol_rate):.2f})")
    legend_handles.append(vio_line)

    if viol_std is not None and len(viol_std) == len(viol_rate):
        ax.fill_between(
            episodes,
            np.array(viol_rate) - np.array(viol_std),
            np.array(viol_rate) + np.array(viol_std),
            color='red',
            alpha=0.2
        )
        legend_handles.append(mpatches.Patch(color='red', alpha=0.2, label='±1 Std Dev (Violations)'))

    # === Final plot formatting ===
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate")
    ax.set_title(f"{title_prefix} – Modification and Violation Rates")
    ax.grid(alpha=0.3)
    ax.legend(handles=legend_handles, loc='best')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"summary_metrics_{timestamp}.png"

    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        save_path = os.path.join(run_dir, filename)
    else:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

    fig.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close(fig)

    def plot_loss_comparison(run_dir, window=10):
        csv_files = glob.glob(os.path.join(run_dir, "train_metrics_*.csv"))
        if not csv_files:
            print(f"No training CSV files found in: {run_dir}")
            return

        all_policy = []
        all_semantic = []
        max_len = 0

        for f in csv_files:
            df = pd.read_csv(f)
            if "policy_loss" not in df.columns or "req_loss" not in df.columns:
                print(f"Missing expected columns in {f}")
                continue

            all_policy.append(df["policy_loss"].values)
            all_semantic.append(df["req_loss"].values)
            max_len = max(max_len, len(df))

        # Pad shorter runs with NaNs so we can compute mean/std
        def pad_runs(runs, length):
            return [np.pad(run, (0, length - len(run)), constant_values=np.nan) for run in runs]

        all_policy = np.array(pad_runs(all_policy, max_len))
        all_semantic = np.array(pad_runs(all_semantic, max_len))

        # Compute means and stds, ignoring NaNs
        policy_mean = np.nanmean(all_policy, axis=0)
        policy_std = np.nanstd(all_policy, axis=0)
        semantic_mean = np.nanmean(all_semantic, axis=0)
        semantic_std = np.nanstd(all_semantic, axis=0)

        steps = np.arange(len(policy_mean))

        # === Plotting ===
        plt.figure(figsize=(10, 6))
        plt.plot(steps, policy_mean, label="Policy Loss", color="blue")
        plt.fill_between(steps, policy_mean - policy_std, policy_mean + policy_std, color="blue", alpha=0.2)

        plt.plot(steps, semantic_mean, label="Semantic Loss", color="green")
        plt.fill_between(steps, semantic_mean - semantic_std, semantic_mean + semantic_std, color="green", alpha=0.2)

        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("PPO Loss vs Semantic Loss (Averaged Across Runs)")
        plt.legend()
        plt.grid(True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(run_dir, f"loss_comparison_{timestamp}.png")
        plt.savefig(save_path)
        plt.show()
        print(f"[Saved] {save_path}")
        plt.close()

def plot_aggregated_training_metrics(run_dir, rolling_window=10):
    """
    Loads all `train_metrics_*.csv` files in `run_dir`, computes average and std across runs,
    and generates reward and constraint plots.
    """
    csv_files = glob.glob(os.path.join(run_dir, "train_metrics_*.csv"))
    if not csv_files:
        print(f"No training CSV files found in: {run_dir}")
        return

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        run_number = os.path.basename(f).split("_")[-1].split(".")[0]
        df["run"] = run_number
        dfs.append(df)

    combined = pd.concat(dfs)

    grouped = combined.groupby("episode").agg({
        "reward": ["mean", "std"],
        "modification_rate": ["mean", "std"],
        "violation_rate": ["mean", "std"],
    }).reset_index()

    # Flatten multi-index columns
    grouped.columns = ["episode", "reward_mean", "reward_std", "mod_mean", "mod_std", "viol_mean", "viol_std"]

    # Plot constraints
    plot_summary_metrics(
        rewards=grouped["reward_mean"].tolist(),
        mod_rate=grouped["mod_mean"].tolist(),
        viol_rate=grouped["viol_mean"].tolist(),
        mod_std=grouped["mod_std"].tolist(),
        viol_std=grouped["viol_std"].tolist(),
        title_prefix="Avg Constraint Metrics, CliffWalking Preemptive-Hard (Across Runs)",
        run_dir=run_dir,
        rolling_window=10
    )

    # plot_rewards(
    #     rewards=grouped["reward_mean"].tolist(),
    #     stds=grouped["reward_std"].to_numpy(),
    #     title="Average Reward Across Runs (CartPole, Preemptive-Hard)",
    #     xlabel="Episode",
    #     ylabel="Reward",
    #     rolling_window=10,
    #     run_dir=run_dir,
    # )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot aggregated training metrics across multiple runs.")
    parser.add_argument("--run_dir", type=str, required=True, help="Directory containing train_metrics_*.csv files")
    parser.add_argument("--rolling_window", type=int, default=1, help="Rolling window size for smoothing")
    args = parser.parse_args()
    plot_aggregated_training_metrics(run_dir=args.run_dir, rolling_window=args.rolling_window)
