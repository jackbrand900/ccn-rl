from functools import partial

import torch
import os
import re
from pishield.propositional_requirements.shield_layer import ShieldLayer as PropositionalShieldLayer
from pishield.linear_requirements.shield_layer import ShieldLayer as LinearShieldLayer

from src.utils.env_helpers import is_in_front_of_key
from src.utils.req_file_to_logic_fn import get_flag_logic_fn

class ShieldController:
    def __init__(self, requirements_path, num_actions, mode="hard", verbose=False, default_flag_logic=None):
        self.requirements_path = requirements_path
        self.num_actions = num_actions
        self.flag_logic_fn = get_flag_logic_fn(self.requirements_path) or self.default_flag_logic
        self.mode = mode
        flag_active_val = 0.8 if mode == "soft" else 1.0
        self.flag_logic_fn = partial(self.flag_logic_fn, flag_active_val=flag_active_val)
        self.flag_logic_batch = self._batchify(self.flag_logic_fn)

        # Parse var names from file
        self.var_names = self._extract_vars_from_requirements()
        self.num_vars = len(self.var_names)

        # Compute action and flag names based on position
        self.action_names = self.var_names[:num_actions]
        self.flag_names = self.var_names[num_actions:]
        self.num_flags = len(self.flag_names)
        self.verbose = verbose

        self.ordering_names = [str(i) for i in range(self.num_vars)]
        self.shield_layer = self.build_shield_layer()
        self.shield_activations = 0

    def _batchify(self, single_fn):
        def batch_fn(contexts):
            return [single_fn(ctx) for ctx in contexts]
        return batch_fn

    def _extract_vars_from_requirements(self):
        with open(self.requirements_path, "r") as f:
            content = f.read()

        # Find all y_i style variables
        vars_found = set(re.findall(r"y_(\d+)", content))
        var_indices = sorted(int(v) for v in vars_found)

        if not var_indices:
            raise ValueError("No variables found in constraints file.")

        max_var = max(var_indices)
        all_vars = [f"y_{i}" for i in range(max_var + 1)]
        return all_vars

    def flag_logic_with_key_check(self, context):
        """
        context: dict with keys 'obs' and 'direction', optionally 'position'
        Returns: dict like {'y_7': 1} if in front of key, assuming y_7 is your flag
        """
        obs = context['obs']  # shape (7,7,3) tensor or np.array
        direction = context['direction']  # 0=right, 1=down, 2=left, 3=up
        in_front = is_in_front_of_key(obs, direction)
        return {"y_7": int(in_front)}  # or whatever your flag var is

    def build_shield_layer(self):
        ordering = ",".join(reversed(self.ordering_names))  # e.g. "3,2,1,0"
        file_ext = os.path.splitext(self.requirements_path)[-1].lower()
        print(f"requirements file: {self.requirements_path}")
        print(f"num vars: {self.num_vars}, num flags: {self.num_flags}")
        if file_ext == ".cnf":
            return PropositionalShieldLayer(
                num_classes=self.num_vars,
                requirements=self.requirements_path,
                ordering_choice="custom",
                custom_ordering=ordering,
            )
        elif file_ext == ".linear":
            return LinearShieldLayer(
                num_variables=self.num_vars,
                requirements_filepath=self.requirements_path,
                ordering_choice="given",
            )
        else:
            raise ValueError(f"Unknown requirements file extension: {file_ext}")

    def default_flag_logic(self, context):
        """Fallback flag logic — always 0 for all flags"""
        return {flag: 0 for flag in self.flag_names}

    def apply(self, action_probs, context):
        flags = self.flag_logic_fn(context)
        flag_values = [flags.get(name, 0) for name in self.flag_names]

        # Expand flags to match batch size
        batch_size = action_probs.size(0)
        flag_tensor = torch.tensor(flag_values, dtype=action_probs.dtype, device=action_probs.device)
        flag_tensor = flag_tensor.unsqueeze(0).expand(batch_size, -1)

        full_input = torch.cat([action_probs, flag_tensor], dim=1)
        shielded_output = self.shield_layer(full_input)
        corrected = shielded_output[:, :self.num_actions]
        if self.mode == "soft":
            corrected = corrected / corrected.sum(dim=1, keepdim=True)

        flag_active = any(flag_values)
        changed = not torch.allclose(action_probs, corrected, atol=1e-5)
        print(f"verbose: {self.verbose}")
        if self.verbose:
            print(f"[DEBUG] Raw flags: {flags}, Flag values: {flag_values}")
            if flag_active:
                print(f"[SHIELD ACTIVE] Flags: {flags}")
                if changed:
                    raw_np = action_probs.detach().cpu().numpy().flatten()
                    corrected_np = corrected.detach().cpu().numpy().flatten()
                    print(f"[SHIELD MODIFIED OUTPUT] Before: {raw_np} → After: {corrected_np}")
                else:
                    print(f"[SHIELD ACTIVE BUT NO CHANGE] Action output remained the same.")

        return corrected

    def apply_batch(self, action_probs, contexts):
        flag_dicts = self.flag_logic_batch(contexts)
        flag_values = [
            [flags.get(name, 0) for name in self.flag_names]
            for flags in flag_dicts
        ]
        flag_tensor = torch.tensor(flag_values, device=action_probs.device, dtype=action_probs.dtype)
        full_input = torch.cat([action_probs, flag_tensor], dim=1)
        shielded_output = self.shield_layer(full_input)
        corrected = shielded_output[:, :self.num_actions]

        if self.mode == "soft":
            corrected = corrected / corrected.sum(dim=1, keepdim=True)

        return corrected

    def forward_differentiable(self, action_probs, contexts):
        assert isinstance(contexts, list), "Contexts must be a list of dicts (batch)."
        flag_dicts = self.flag_logic_batch(contexts)

        flag_values = [
            [flags.get(name, 0.0) for name in self.flag_names]
            for flags in flag_dicts
        ]
        flag_tensor = torch.tensor(
            flag_values, dtype=action_probs.dtype, device=action_probs.device
        )  # shape: [B, num_flags]

        full_input = torch.cat([action_probs, flag_tensor], dim=1)
        shielded_output = self.shield_layer(full_input)
        corrected = shielded_output[:, :self.num_actions]

        if self.mode == "soft":
            corrected = corrected / corrected.sum(dim=1, keepdim=True)

        # Count activations
        modified = ~torch.isclose(action_probs, corrected, atol=1e-5)
        activated = modified.any(dim=1)  # shape [B]
        self.shield_activations += activated.sum().item()

        return corrected

    def would_violate(self, selected_action, context):
        """
        Checks whether the selected action violates the constraints.
        Returns 1 if it does, 0 if not.
        """
        one_hot = torch.zeros(1, self.num_actions, dtype=torch.float32)
        one_hot[0, selected_action] = 1.0

        # Generate flags
        flags = self.flag_logic_fn(context)
        flag_values = [flags.get(name, 0) for name in self.flag_names]
        flag_tensor = torch.tensor(flag_values, dtype=one_hot.dtype).unsqueeze(0)

        # Concatenate and check what shield outputs
        input_tensor = torch.cat([one_hot, flag_tensor], dim=1)
        corrected = self.shield_layer(input_tensor)
        corrected_action = corrected[0, :self.num_actions].argmax().item()
        return int(corrected_action != selected_action)
