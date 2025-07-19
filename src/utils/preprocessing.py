import torch
import numpy as np

def prepare_input(state, use_cnn=False):
    state_tensor = torch.FloatTensor(state)

    if use_cnn:
        # Fix for flat image accidentally passed in
        if state_tensor.ndim == 1 and state_tensor.numel() == 96 * 96 * 3:
            state_tensor = state_tensor.view(96, 96, 3)

        if state_tensor.ndim == 3 and state_tensor.shape[-1] == 3:
            state_tensor = state_tensor.permute(2, 0, 1)  # (HWC) → (CHW)

        state_tensor = state_tensor.unsqueeze(0) / 255.0  # Normalize and add batch dim
    else:
        state_tensor = state_tensor.flatten().unsqueeze(0)

    return state_tensor

def prepare_batch(states, use_cnn=False):
    """
    Prepare a batch of states for model input.
    Expects a list of np.ndarrays or torch tensors.
    """
    if isinstance(states[0], torch.Tensor):
        states = [s.detach().cpu().numpy() for s in states]

    states = np.array(states)

    # Defensive check: if input is [B, D] shape, it's already flattened (bad for CNN)
    if use_cnn:
        if states.ndim == 2:
            raise ValueError(f"Expected image-shaped input, got flat tensor shape: {states.shape}")

        if states.ndim == 4 and states.shape[-1] == 3:  # [B, H, W, C]
            states_tensor = torch.FloatTensor(states).permute(0, 3, 1, 2) / 255.0
        elif states.ndim == 4 and states.shape[1] == 3:  # [B, C, H, W] already
            states_tensor = torch.FloatTensor(states) / 255.0
        else:
            raise ValueError(f"Unexpected CNN input shape: {states.shape}")
    else:
        states_tensor = torch.FloatTensor(states).view(len(states), -1)

    return states_tensor
