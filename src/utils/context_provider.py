import numpy as np

import src.utils.env_helpers as env_helpers


def build_context(env, agent):
    env_id = env.spec.id if hasattr(env, "spec") and env.spec else ""
    env_unwrapped = env.unwrapped
    context = {
        "timestep": getattr(agent, "learn_step_counter", 0),
        "env": env,
    }

    if "MiniGrid" in env_id:
        position = env_helpers.extract_agent_pos(env)
        direction = env.unwrapped.agent_dir
        obs_dict = env.unwrapped.gen_obs()
        obs = obs_dict["image"]

        context.update({
            "position": position,
            "direction": direction,
            "obs": obs,
        })

    elif "CarRacing" in env_id:
        obs = agent.last_obs if hasattr(agent, "last_obs") else None
        context["obs"] = obs

        # Red light flag
        if hasattr(env_unwrapped, "red_light_active"):
            context["at_red_light"] = env_unwrapped.red_light_active

        if hasattr(env_unwrapped, "wheel_on_grass"):
            # Store each wheel's grass status as flags y_6–y_9
            for i, on_grass in enumerate(env_unwrapped.wheel_on_grass):
                context[f"wheel_{i}_on_grass"] = on_grass
    else:
        obs = agent.last_obs if hasattr(agent, "last_obs") else None
        # print(f'obs: {obs}')
        context["obs"] = obs

    return context


def position_flag_logic(context, flag_active_val=1.0):
    pos = context.get("position", None)
    return {"y_7": flag_active_val if (pos == (1, 1)) else 0}


def key_flag_logic(context, flag_active_val=1.0):
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
        0: (1, 0),  # right
        1: (0, 1),  # down
        2: (-1, 0),  # left
        3: (0, -1),  # up
    }
    dx, dy = front_deltas[direction]
    front_pos = (position[0] + dx, position[1] + dy)

    key_at_front = (key_pos == front_pos)

    # print(f"[FLAG] Agent @ {position}, facing {direction}, front pos {front_pos}")
    # print(f"[FLAG] Key @ {key_pos} — Exists: {key_obj is not None}")
    # print(f"[FLAG] Grid check says key_at_front = {key_at_front}")

    return {"y_7": flag_active_val if key_at_front else 0}


def cartpole_flag_logic(context):
    obs = context.get("obs")
    if obs is None:
        return {}

    pos, vel, angle, angle_vel = obs

    def soft_flag(x, center, width=0.8):
        return float(np.clip((x - center) / width, 0, 1))

    return {
        "y_2": soft_flag(pos, 0.8),  # cart right
        "y_3": soft_flag(-pos, 0.8),  # cart left
        "y_4": soft_flag(angle if angle_vel > 0 else 0, 0.05),  # pole falling right
        "y_5": soft_flag(-angle if angle_vel < 0 else 0, 0.05),  # pole falling left
    }


def cartpole_flag_logic_advanced(context,
                                 theta_thresh=0.05,
                                 theta_scale=0.5,
                                 pos_thresh=0.8,
                                 pos_scale=0.5,
                                 emergency_thresh=0.15,
                                 emergency_scale=0.15):
    obs = context.get("obs")
    if obs is None:
        return {}

    pos, vel, angle, angle_vel = obs

    def ramp(x, threshold, scale):
        return float(np.clip((x - threshold) / scale, 0, 1))

    flags = {
        "y_2": ramp(angle, theta_thresh, theta_scale) * ramp(angle_vel, 0.5, 0.5),  # falling right
        "y_3": ramp(-angle, theta_thresh, theta_scale) * ramp(-angle_vel, 0.5, 0.5),  # falling left
        "y_4": ramp(pos, pos_thresh, pos_scale) * ramp(vel, 0.5, 0.5),  # cart right
        "y_5": ramp(-pos, pos_thresh, pos_scale) * ramp(-vel, 0.5, 0.5),  # cart left
        "y_6": ramp(abs(angle), emergency_thresh, emergency_scale),  # emergency
    }

    return flags


def cartpole_emergency_flag_logic(context, flag_active_val=1.0):
    # print("[DEBUG] cartpole_emergency_flag_logic called with context:", context)

    obs = context.get("obs")
    if obs is None:
        return {}

    _, _, angle, _ = obs

    return {
        "y_2": flag_active_val if angle < -0.15 else 0.0,  # tipping far to the left
        "y_3": flag_active_val if angle > 0.15 else 0.0,  # tipping far to the right
    }


def red_light_flag_logic(context, flag_active_val=1.0):
    at_red_light = context.get("at_red_light", False)

    return {
        "y_5": flag_active_val if at_red_light else 0.0
    }


def wheel_on_grass_flag_logic(context, flag_active_val=1.0):
    return {
        "y_5": flag_active_val if context.get("wheel_0_on_grass", False) else 0.0,
        "y_6": flag_active_val if context.get("wheel_1_on_grass", False) else 0.0,
        "y_7": flag_active_val if context.get("wheel_2_on_grass", False) else 0.0,
        "y_8": flag_active_val if context.get("wheel_3_on_grass", False) else 0.0,
    }