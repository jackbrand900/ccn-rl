def config_by_env(env_name, use_ram_obs=False):
    if "CarRacing" in env_name:
        return {
            "use_cnn": True,
            "input_shape": (96, 96, 3),
            "frame_stack": 1,
            "max_steps": 25,
        }

    if "Freeway" in env_name:
        if use_ram_obs:
            print('using RAM')
            return {
                "use_cnn": False,
                "input_shape": (128,),
                "frame_stack": 1,
                "max_steps": 1000,
            }
        else:
            return {
                "use_cnn": True,
                "input_shape": (1, 84, 84),
                "frame_stack": 1,
                "max_steps": 1000,
            }
    if "Seaquest" in env_name:
        if use_ram_obs:
            print('using RAM')
            return {
                "use_cnn": False,
                "input_shape": (128,),
                "frame_stack": 1,
                "max_steps": 10000,
            }
        else:
            return {
                "use_cnn": True,
                "input_shape": (1, 84, 84),
                "frame_stack": 1,
                "max_steps": 10000,
            }
    if "DemonAttack" in env_name:
        if use_ram_obs:
            print('using RAM')
            return {
                "use_cnn": False,
                "input_shape": (128,),
                "frame_stack": 1,
                "max_steps": 10000,
            }
        else:
            return {
                "use_cnn": True,
                "input_shape": (1, 84, 84),
                "frame_stack": 1,
                "max_steps": 10000,
            }
    if "MiniGrid" in env_name:
        return {
            "use_cnn": False,
            "input_shape": 147,  # FlatObsWrapper produces flat vector
            "frame_stack": 1,
            "max_steps": 200,
        }
    if "CartPole" in env_name:
        return {
            "use_cnn": False,
            "input_shape": 4,
            "frame_stack": 1,
            "max_steps": 500,
        }
    if "CliffWalking" in env_name:
        return {
            "use_cnn": False,
            "input_shape": 4,
            "frame_stack": 1,
            "max_steps": 500,
        }
    if "LunarLander" in env_name:
        return {
            "use_cnn": False,
            "input_shape": 8,
            "frame_stack": 1,
            "max_steps": 1000,
        }
    if "Acrobot" in env_name:
        return {
            "use_cnn": False,
            "input_shape": 6,
            "frame_stack": 1,
            "max_steps": 500,
        }
    if "FrozenLake" in env_name:
        return {
            "use_cnn": False,
            "input_shape": 16,
            "frame_stack": 1,
            "max_steps": 100,
        }
    if "Taxi" in env_name:
        return {
            "use_cnn": False,
            "input_shape": 500,
            "frame_stack": 1,
            "max_steps": 200,
        }
    raise ValueError(f"Unknown environment: {env_name}")

