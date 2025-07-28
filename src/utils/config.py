def config_by_env(env_name):
    if "CarRacing" in env_name:
        return {
            "use_cnn": True,
            "input_shape": (96, 96, 3),
            "frame_stack": 1,
            "max_steps": 25,
        }
    elif env_name == "FreewayNoFrameskip-v4":
        return {
            "use_cnn": False,
            "input_shape": 128,
            "frame_stack": 1,
            "max_steps": 1000,
        }
    elif env_name == "ALE/Freeway-v5":
        # return {
        #     "use_cnn": False,
        #     "input_shape": 26,
        #     "frame_stack": 1,
        #     "max_steps": 1000,
        # }
        return {
            "use_cnn": True,
            "input_shape": (1, 84, 84),
            "frame_stack": 1,
            "max_steps": 25,
        }
    elif "Freeway" in env_name:
        return {
            "use_cnn": True,
            "input_shape": (96, 96, 3),
            "frame_stack": 1,
            "max_steps": 25,
        }
    elif "Seaquest" in env_name:
        return {
            "use_cnn": True,
            "input_shape": (96, 96, 3),
            "frame_stack": 1,
            "max_steps": 25,
        }
        # return {
        #     "use_cnn": False,
        #     "input_shape": 15,
        #     "frame_stack": 1,
        #     "max_steps": 5000,
        # }
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

