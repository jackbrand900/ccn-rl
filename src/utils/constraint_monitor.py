import torch

class ConstraintMonitor:
    def __init__(self, only_if_flags_active=True, verbose=False):
        self.only_if_flags_active = only_if_flags_active
        self.verbose = verbose
        self.episode_steps = 0
        self.episode_modifications = 0
        self.episode_violations = 0
        self.episode_flagged_steps = 0
        self.total_steps = 0
        self.total_modifications = 0
        self.total_violations = 0
        self.total_flagged_steps = 0
        self.reset()

    def reset(self):
        self.episode_steps = 0
        self.episode_modifications = 0
        self.episode_violations = 0
        self.episode_flagged_steps = 0

    def log_step_from_probs_and_actions(
            self,
            raw_probs,
            corrected_probs,
            a_unshielded,
            a_shielded,
            context=None,
            shield_controller=None,
            epsilon=1e-6,
    ):
        self.episode_steps += 1
        self.total_steps += 1

        # === Check if probabilities changed ===
        probs_modified = not torch.allclose(raw_probs, corrected_probs, atol=epsilon)
        modification = int(a_unshielded != a_shielded)

        # === Check true violation: would the selected action have been changed? ===
        violation = False
        if shield_controller is not None:
            violation = self.would_violate(
                action=a_shielded if shield_controller.is_shield_active else a_unshielded,
                context=context,
                shield_controller=shield_controller
            )

        # === Flag check ===
        flag_active = True
        if self.only_if_flags_active and context is not None and shield_controller is not None:
            flags = shield_controller.flag_logic_fn(context)
            flag_values = [flags.get(name, 0) for name in shield_controller.flag_names]
            flag_active = any(flag_values)
            if flag_active:
                self.episode_flagged_steps += 1
                self.total_flagged_steps += 1

        if flag_active:
            self.episode_violations += int(violation)
            self.total_violations += int(violation)

            if shield_controller is not None and shield_controller.is_shield_active:
                self.episode_modifications += modification
                self.total_modifications += modification

        if self.verbose and (not self.only_if_flags_active or flag_active):
            print(f"[ConstraintMonitor] Flags active: {flag_active}, "
                  f"Mod: {modification}, Viol: {violation}, "
                  f"Probs Modified: {probs_modified}")

    def would_violate(self, action, context, shield_controller):
        """
        Returns True if the given action would be changed by the shield layer.
        """
        device = next(shield_controller.shield_layer.parameters()).device  # get device from shield layer

        one_hot = torch.zeros(1, shield_controller.num_actions, dtype=torch.float32, device=device)
        one_hot[0, action] = 1.0

        # Get current flags
        flags = shield_controller.flag_logic_fn(context)
        flag_values = [flags.get(name, 0) for name in shield_controller.flag_names]
        flag_tensor = torch.tensor(flag_values, dtype=one_hot.dtype, device=device).unsqueeze(0)

        input_tensor = torch.cat([one_hot, flag_tensor], dim=1)
        corrected = shield_controller.shield_layer(input_tensor)
        corrected_action = corrected[0, :shield_controller.num_actions].argmax().item()

        return int(corrected_action != action)

    def summary(self):
        return {
            # Episode-level metrics
            "episode_steps": self.episode_steps,
            "episode_flagged_steps": self.episode_flagged_steps,
            "episode_modifications": self.episode_modifications,
            "episode_violations": self.episode_violations,
            "episode_mod_rate": self.episode_modifications / max(self.episode_steps, 1),
            "episode_viol_rate": self.episode_violations / max(self.episode_steps, 1),

            # Total metrics
            "total_steps": self.total_steps,
            "total_flagged_steps": self.total_flagged_steps,
            "total_modifications": self.total_modifications,
            "total_violations": self.total_violations,
            "total_mod_rate": self.total_modifications / max(self.total_steps, 1),
            "total_viol_rate": self.total_violations / max(self.total_steps, 1),
        }
