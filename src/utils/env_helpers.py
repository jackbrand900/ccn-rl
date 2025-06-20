def extract_agent_pos(flat_obs):
    pos = flat_obs.unwrapped.agent_pos
    return pos

def convert_action_to_string(action):
    action_names = [
        "left",    # 0
        "right",   # 1
        "forward", # 2
        "pickup",  # 3
        "drop",    # 4
        "toggle",  # 5
        "done"     # 6
    ]
    if 0 <= action < len(action_names):
        return action_names[action]
    return f"unknown({action})"