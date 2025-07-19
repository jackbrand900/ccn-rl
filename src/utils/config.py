def config_by_env(env_name):
    if "CarRacing" in env_name:
        return {
            "use_cnn": True,
            "input_shape": (3, 64, 64),
            "frame_stack": 1,
            "max_steps": 500,
        }
    elif "MiniGrid" in env_name:
        return {
            "use_cnn": False,
            "input_shape": 147,  # FlatObsWrapper produces flat vector
            "frame_stack": 1,
            "max_steps": 200,
        }
    elif "CartPole" in env_name:
        return {
            "use_cnn": False,
            "input_shape": 4,
            "frame_stack": 1,
            "max_steps": 500,
        }
    else:
        raise ValueError(f"Unknown environment: {env_name}")
