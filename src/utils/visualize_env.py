import matplotlib.pyplot as plt

_visualizers = {}

def register_visualizer(env_name):
    def decorator(func):
        _visualizers[env_name] = func
        return func
    return decorator

def visualize_env(env, title="Environment"):
    # Get environment ID or type for selecting visualizer
    env_name = None
    if hasattr(env, 'spec') and env.spec is not None:
        env_name = env.spec.id
    elif hasattr(env, 'unwrapped'):
        env_name = type(env.unwrapped).__name__
    else:
        env_name = type(env).__name__

    visualizer = _visualizers.get(env_name, default_visualizer)
    visualizer(env, title=title)

def unwrap_env(env):
    # Recursively unwrap env wrappers until the base env is found
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    return base_env

def default_visualizer(env, title="Environment"):
    try:
        base_env = unwrap_env(env)
        img = base_env.render(mode='rgb_array')
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Default visualizer failed: {e}")
        print("No visualization available for this environment.")

@register_visualizer("MiniGrid-Empty-5x5-v0")
def visualize_minigrid(env, title="MiniGrid Environment"):
    base_env = unwrap_env(env)
    try:
        img = base_env.render(mode='rgb_array')
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Visualization failed: {e}")

    pos = base_env.agent_pos
    cell = base_env.grid.get(*pos)
    color = getattr(cell, 'color', None) if cell else None
    print(f"Agent pos: {pos}, Cell color: {color}")
