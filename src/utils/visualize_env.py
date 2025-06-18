import matplotlib.pyplot as plt
import numpy as np
import math
from collections import defaultdict

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