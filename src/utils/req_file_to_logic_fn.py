from src.utils.context_provider import (key_flag_logic, position_flag_logic, cartpole_emergency_flag_logic,
                                        red_light_flag_logic, wheel_on_grass_flag_logic)

req_file_to_logic_fn = {
    "emergency_cartpole.cnf": cartpole_emergency_flag_logic,
    "forward_on_flag.cnf": position_flag_logic,
    "key_on_flag.cnf": key_flag_logic,
    "red_light_stop.cnf": red_light_flag_logic,
    "wheel_on_grass.cnf": wheel_on_grass_flag_logic
}


def get_flag_logic_fn(requirements_path):
    file_name = requirements_path.split("/")[-1]
    if file_name not in req_file_to_logic_fn:
        raise ValueError(f"Unsupported requirements file: {requirements_path}")
    return req_file_to_logic_fn[file_name]
