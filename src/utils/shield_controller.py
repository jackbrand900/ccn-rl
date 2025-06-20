import torch
from pishield.propositional_requirements.shield_layer import ShieldLayer as PropositionalShieldLayer
from pishield.linear_requirements.shield_layer import ShieldLayer as LinearShieldLayer

class ShieldController:
    def __init__(self, requirements_path, num_actions, num_flags=1, flag_logic_fn=None, threshold=0.5):
        self.requirements_path = requirements_path
        self.num_actions = num_actions
        self.num_flags = num_flags
        self.flag_logic_fn = flag_logic_fn or self.default_flag_logic
        self.flag_logic_batch = self._batchify(self.flag_logic_fn)

        # Generate var names: 0 to {total_vars-1}
        self.var_names = [f"y_{i}" for i in range(num_actions + num_flags)]
        self.ordering_names = [f"{i}" for i in range(num_actions + num_flags)]
        self.num_vars = len(self.var_names)
        self.action_names = self.var_names[:num_actions]
        self.flag_names = self.var_names[num_actions:]
        self.shield_layer = self.build_shield_layer()

    def _batchify(self, single_fn):
        def batch_fn(contexts):
            return [single_fn(ctx) for ctx in contexts]
        return batch_fn

    def build_shield_layer(self):
        ordering = ",".join(reversed(self.ordering_names))  # e.g. 3,2,1,0
        # TODO: make this dynamic based on the type of shield layer (propositional or linear)
        return PropositionalShieldLayer(
            num_classes=self.num_vars,
            requirements=self.requirements_path,
            ordering_choice="custom",
            custom_ordering=ordering,
        )
        # return LinearShieldLayer(
        #     num_variables=self.num_vars,
        #     requirements_filepath=self.requirements_path,
        #     ordering_choice="given"
        # )

    def default_flag_logic(self, context):
        """Fallback flag logic — always 0 for all flags"""
        return {flag: 0 for flag in self.flag_names}

    def apply(self, action_probs, context, verbose=False):
        # === Apply flag logic ===
        flags = self.flag_logic_fn(context)
        flag_values = [flags.get(name, 0) for name in self.flag_names]

        # Expand flags to match batch size
        batch_size = action_probs.size(0)
        flag_tensor = torch.tensor(flag_values, dtype=action_probs.dtype, device=action_probs.device)
        flag_tensor = flag_tensor.unsqueeze(0).expand(batch_size, -1)

        # === Concatenate actions and flags ===
        full_input = torch.cat([action_probs, flag_tensor], dim=1)

        # === Apply shield ===
        shielded_output = self.shield_layer(full_input)

        # === Mask out flags and return corrected action distribution ===
        corrected = shielded_output[:, :self.num_actions]

        # === Optional debug output ===
        flag_active = any(flag_values)
        changed = not torch.allclose(action_probs, corrected, atol=1e-5)
        if verbose:
            position = context.get("position", "N/A")
            print(f"Position: {position}")
            print(f"[DEBUG] Raw flags: {flags}, Flag values: {flag_values}")
            if flag_active:
                print(f"[SHIELD ACTIVE] Flags: {flags}")
                if changed:
                    print(f"[SHIELD MODIFIED OUTPUT] Before: {action_probs.cpu().numpy().flatten()} → After: {corrected.cpu().numpy().flatten()}")
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
        return shielded_output[:, :self.num_actions]
