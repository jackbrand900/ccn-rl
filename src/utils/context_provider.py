import src.utils.env_helpers as env_helpers

def build_context(env, agent):
    position = env_helpers.extract_agent_pos(env)
    timestep = agent.learn_step_counter
    return {
        "position": position,
        "timestep": timestep,
    }

def position_flag_logic(context):
    pos = context.get("position", None)
    return {"y_7": int(pos == (1, 1))}

