import src.utils.env_helpers as env_helpers


def build_context(env, agent):
    position = env_helpers.extract_agent_pos(env)
    timestep = agent.learn_step_counter

    obs_dict = env.unwrapped.gen_obs()
    obs = obs_dict["image"]
    direction = env.unwrapped.agent_dir

    return {
        "position": position,
        "timestep": timestep,
        "obs": obs,
        "direction": direction,
        "env": env,
    }


def position_flag_logic(context):
    pos = context.get("position", None)
    return {"y_7": int(pos == (1, 1))}


def key_flag_logic(context):
    direction = context.get("direction")
    position = context.get("position")
    env = context.get("env")

    if None in (direction, position, env) or not hasattr(env, "key_pos"):
        print("[FLAG] Missing direction, position, env, or env.key_pos")
        return {"y_7": 0}

    key_pos = env.key_pos

    # Check if the key still exists in the grid
    key_obj = env.unwrapped.grid.get(*key_pos)
    if key_obj is None or key_obj.type != "key":
        # print(f"[FLAG] Key no longer exists at {key_pos}")
        return {"y_7": 0}

    # Compute global front position
    front_deltas = {
        0: (1, 0),   # right
        1: (0, 1),   # down
        2: (-1, 0),  # left
        3: (0, -1),  # up
    }
    dx, dy = front_deltas[direction]
    front_pos = (position[0] + dx, position[1] + dy)

    key_at_front = (key_pos == front_pos)

    # print(f"[FLAG] Agent @ {position}, facing {direction}, front pos {front_pos}")
    # print(f"[FLAG] Key @ {key_pos} — Exists: {key_obj is not None}")
    # print(f"[FLAG] Grid check says key_at_front = {key_at_front}")

    return {"y_7": int(key_at_front)}