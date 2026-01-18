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
    elif "Freeway" in env_id:
        # Extract symbolic features directly from RAM regardless of observation mode
        context["obs"] = agent.last_obs if hasattr(agent, "last_obs") else None

        try:
            ram = env.unwrapped.ale.getRAM()
            player_y = ram[35] / 255.0
            crossings = ram[83] / 255.0

            # Car positions (RAM[40–47], and 48–55 as secondary)
            num_lanes = 8
            car_positions = []
            for lane in range(num_lanes):
                base = 40 + lane
                x1 = ram[base] / 160.0
                x2 = ram[(base + 8) % 128] / 160.0
                car_positions.extend([x1, x2])

            # Estimate car speeds
            car_speeds = []
            for lane in range(num_lanes):
                curr = ram[40 + lane] / 160.0
                prev = getattr(env, "_prev_car_positions", np.zeros((num_lanes,), dtype=np.float32))[lane]
                speed = np.clip((curr - prev) * 10, -1.0, 1.0)
                car_speeds.append(speed)

            # Cache current positions for next frame
            env._prev_car_positions = np.array([ram[40 + i] / 160.0 for i in range(num_lanes)], dtype=np.float32)

            # Compute lane
            player_lane = min(max(int(player_y * 8), 0), 7)
            car_xs = car_positions[player_lane * 2 : player_lane * 2 + 2]
            car_near = any(abs(x - 0.5) < 0.15 for x in car_xs)

            if player_lane < 7:
                lane_ahead = player_lane + 1
                car_xs_ahead = car_positions[lane_ahead * 2 : lane_ahead * 2 + 2]
                car_ahead_near = any(abs(x - 0.5) < 0.15 for x in car_xs_ahead)
            else:
                car_ahead_near = False

            safe_to_go_up = not car_ahead_near

            context.update({
                "player_y": player_y,
                "crossings": crossings,
                "player_lane": player_lane,
                "car_near": car_near,
                "car_ahead_near": car_ahead_near,
                "safe_to_go_up": safe_to_go_up,
                "car_speeds": car_speeds,
            })

        except Exception as e:
            # fallback
            context["safe_to_go_up"] = True

    elif "Seaquest" in env_id:
        try:
            ram = env.unwrapped.ale.getRAM()

            context["obs"] = agent.last_obs if hasattr(agent, "last_obs") else None

            # === Player position ===
            player_x = ram[70]
            player_y = ram[97]
            low_depth = player_y > 70

            # === Oxygen status ===
            oxygen_meter = ram[102]
            low_oxygen = oxygen_meter < 20

            # === Player direction ===
            player_direction = ram[86]
            facing_right = player_direction == 1
            facing_left = player_direction == 2

            # === Divers rescued ===
            divers_collected = ram[62]

            context.update({
                "player_x": player_x,
                "player_y": player_y,
                "oxygen": oxygen_meter,
                "low_oxygen": low_oxygen,
                "low_depth": low_depth,
                "facing_right": facing_right,
                "facing_left": facing_left,
                "divers_collected": divers_collected,
            })

        except Exception as e:
            print(f"[Seaquest RAM error] {e}")
            context["oxygen"] = 1.0
    elif "DemonAttack" in env_id:
        context["obs"] = agent.last_obs if hasattr(agent, "last_obs") else None

        try:
            ram = env.unwrapped.ale.getRAM()

            player_x = ram[22] / 255.0
            enemy_xs = [ram[i] / 255.0 for i in [17, 18, 19]]
            enemy_ys = [ram[i] / 255.0 for i in [69, 70, 71]]
            missile_y = ram[21] / 255.0
            level = ram[62]
            num_lives = ram[114]

            context.update({
                "player_x": player_x,
                "enemy_xs": enemy_xs,
                "enemy_ys": enemy_ys,
                "missile_y": missile_y,
                "level": level,
                "num_lives": num_lives,
            })
        except Exception as e:
            print(f"[DemonAttack RAM error] {e}")
    elif "CliffWalking" in env_id:
        obs = agent.last_obs if hasattr(agent, "last_obs") else None
        context["obs"] = obs

        try:
            state = int(np.argmax(obs)) if isinstance(obs, np.ndarray) else int(obs)
            width = 12
            height = 4

            x = state % width
            y = state // width

            def is_cliff(s):
                # Cliff is the bottom row excluding start (36) and goal (47)
                return s in list(range(37, 47))

            cliff_right = False
            cliff_down = False

            # Check if moving RIGHT leads to cliff
            if x < width - 1:
                s_right = state + 1
                cliff_right = is_cliff(s_right)

            # Check if moving DOWN leads to cliff
            if y < height - 1:
                s_down = state + width
                cliff_down = is_cliff(s_down)

            context["cliff_right"] = cliff_right
            context["cliff_down"] = cliff_down
            # print(f'state: {state}')
            # print(f'cliff right? {cliff_right}')
            # print(f'cliff down? {cliff_down}')

        except Exception as e:
            print(f"[CliffWalking context error] {e}")
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


def freeway_flag_logic(context, flag_active_val=1.0):
    return {
        "y_3": flag_active_val if context.get("car_ahead_near", False) else 0.0,
        "y_4": flag_active_val if context.get("safe_to_go_up", False) else 0.0,
        "y_5": flag_active_val if context.get("car_near", False) else 0.0,
    }

def seaquest_flag_logic(context, flag_active_val=1.0):
    return {
        "y_18": flag_active_val if context.get("low_oxygen", False) else 0.0,        # Low oxygen → must surface
        "y_19": flag_active_val if context.get("divers_collected", 0) > 0 else 0.0,  # at least one diver collected
        "y_20": flag_active_val if context.get("facing_right", False) else 0.0,      # Facing right
        "y_21": flag_active_val if context.get("facing_left", False) else 0.0,       # Facing left
        "y_22": flag_active_val if context.get("low_depth", False) else 0.0
    }

def demonattack_flag_logic(context, flag_active_val=1.0):
    flags = {}

    # === Context extraction ===
    player_x = context.get("player_x", 0.5)
    enemy_xs = context.get("enemy_xs", [])
    enemy_ys = context.get("enemy_ys", [])
    missile_y = context.get("missile_y", 0.5)

    # === THREAT FLAGS ===

    # y_6: Enemy very close to bottom (immediate danger)
    flags["y_6"] = flag_active_val if any(ey > 0.85 for ey in enemy_ys) else 0.0

    # y_7: Missile is still in flight (can't fire again yet)
    flags["y_7"] = flag_active_val if missile_y > 0.2 else 0.0

    # y_8: Any enemy decently aligned with the player (within 0.1)
    aligned = any(abs(ex - player_x) < 0.1 for ex in enemy_xs)
    flags["y_8"] = flag_active_val if aligned else 0.0

    # y_10: High-density wave (many enemies visible)
    flags["y_10"] = flag_active_val if len(enemy_xs) >= 3 else 0.0

    return flags

def cliffwalking_flag_logic(context, flag_active_val=1.0):
    cliff_right = context.get("cliff_right", False)
    cliff_down = context.get("cliff_down", False)

    # print(f"[Flag Logic] cliff_right={cliff_right}, cliff_down={cliff_down}")

    return {
        "y_4": flag_active_val if cliff_right else 0.0,
        "y_5": flag_active_val if cliff_down else 0.0,
    }



