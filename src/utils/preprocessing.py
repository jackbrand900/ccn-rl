import torch
import numpy as np

def prepare_input(state, use_cnn=False):
    state_tensor = torch.FloatTensor(np.array(state))

    if use_cnn:
        if state_tensor.ndim == 3:
            # Handle (H, W, C) → (C, H, W)
            if state_tensor.shape[-1] == 3:
                state_tensor = state_tensor.permute(2, 0, 1)
            # If already (C, H, W), do nothing
        elif state_tensor.ndim == 4:
            # Handle (stack, H, W, C) → (stack * C, H, W)
            if state_tensor.shape[-1] == 3:
                stack, h, w, c = state_tensor.shape
                state_tensor = state_tensor.permute(0, 3, 1, 2).reshape(stack * c, h, w)
            else:
                # (stack, C, H, W) → (stack * C, H, W)
                stack, c, h, w = state_tensor.shape
                state_tensor = state_tensor.reshape(stack * c, h, w)
        else:
            raise ValueError(f"Expected image with shape (C, H, W), (H, W, C), (stack, H, W, C), or (stack, C, H, W), got {state_tensor.shape}")

        state_tensor = state_tensor.unsqueeze(0) / 255.0  # Normalize image data
    else:
        # 🛠️ Normalize RAM data to [0, 1]
        state_tensor = state_tensor.flatten().unsqueeze(0) / 255.0

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
                # Already [B, C, H, W]
                states_tensor = torch.FloatTensor(states) / 255.0
        elif states.ndim == 5:
            # [B, stack, H, W, C] → [B, stack * C, H, W]
            if states.shape[-1] == 3:
                B, stack, H, W, C = states.shape
                states_tensor = torch.FloatTensor(states).permute(0, 1, 4, 2, 3).reshape(B, stack * C, H, W) / 255.0
            else:
                # [B, stack, C, H, W] → [B, stack * C, H, W]
                B, stack, C, H, W = states.shape
                states_tensor = torch.FloatTensor(states).reshape(B, stack * C, H, W) / 255.0
        else:
            raise ValueError(f"Unexpected image shape: {states.shape}")
    else:
        # 🛠️ Normalize RAM batch to [0, 1]
        states_tensor = torch.FloatTensor(states).view(len(states), -1) / 255.0

    return states_tensor
