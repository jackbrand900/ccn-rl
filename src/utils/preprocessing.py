import torch
import numpy as np

def prepare_input(state, use_cnn=False):
    state_tensor = torch.FloatTensor(state)

    if use_cnn:
        if state_tensor.ndim == 3:
            # Handle (H, W, C) → (C, H, W)
            if state_tensor.shape[-1] == 3:
                state_tensor = state_tensor.permute(2, 0, 1)
            # If already (C, H, W), do nothing
        elif state_tensor.ndim == 4:
            raise ValueError("Expected single state (not batch) with shape (C, H, W) or (H, W, C)")

        state_tensor = state_tensor.unsqueeze(0) / 255.0  # Add batch and normalize
    else:
        state_tensor = state_tensor.flatten().unsqueeze(0)

    return state_tensor

def prepare_batch(states, use_cnn=False):
    if isinstance(states[0], torch.Tensor):
        states = [s.detach().cpu().numpy() for s in states]

    states = np.array(states)

    if use_cnn:
        if states.ndim == 4:
            # [B, H, W, C] → [B, C, H, W]
            if states.shape[-1] == 3:
                states_tensor = torch.FloatTensor(states).permute(0, 3, 1, 2) / 255.0
            else:
                # Already [B, C, H, W] — stacked grayscale
                states_tensor = torch.FloatTensor(states) / 255.0
        else:
            raise ValueError(f"Unexpected image shape: {states.shape}")
    else:
        states_tensor = torch.FloatTensor(states).view(len(states), -1)

    return states_tensor