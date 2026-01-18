import numpy as np

def extract_agent_pos(flat_obs):
    pos = flat_obs.unwrapped.agent_pos
    return pos


def convert_action_to_string(action):
    action_names = [
        "left",  # 0
        "right",  # 1
        "forward",  # 2
        "pickup",  # 3
        "drop",  # 4
        "toggle",  # 5
        "done"  # 6
    ]
    if 0 <= action < len(action_names):
        return action_names[action]
    return f"unknown({action})"


def get_agent_view(env):
    grid, vis_mask = env.unwrapped.gen_obs_grid()
    encoded = grid.encode()  # this is a list-of-lists of tuples
    return np.array(encoded, dtype=np.int32)  # shape: (7, 7, 3)


def is_in_front_of_key(obs_tensor, direction):
    """
    Checks the agent-centered 7x7 observation for a key directly in front of the agent.
    Assumes obs_tensor has shape (7, 7, 3).
    """
    front_coords = {
        0: (3, 2),  # facing right
        1: (4, 3),  # facing down
        2: (3, 4),  # facing left
        3: (2, 3),  # facing up
    }

    fx, fy = front_coords[direction]

    # Ensure obs_tensor is a NumPy array
    import numpy as np
    if not isinstance(obs_tensor, np.ndarray):
        obs_tensor = np.array(obs_tensor)

    cell = obs_tensor[fx, fy]
    object_type = cell[0]
    print(f"[OBS] obs[{fx}, {fy}] = {cell}")
    print(f"[OBS] At view({fx},{fy}) → object_type: {object_type} (should be 5 for key)")
    return int(object_type == 5)

def find_key(env):
    grid = env.unwrapped.grid
    for x in range(grid.width):
        for y in range(grid.height):
            obj = grid.get(x, y)
            if obj is not None and obj.type == "key":
                return (x, y)
    return None