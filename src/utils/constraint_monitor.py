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

        # Flag check
        flag_active = True
        if self.only_if_flags_active and context is not None and shield_controller is not None:
            flags = shield_controller.flag_logic_fn(context)
            flag_values = [flags.get(name, 0) for name in shield_controller.flag_names]
            flag_active = any(flag_values)
            if flag_active:
                self.episode_flagged_steps += 1
                self.total_flagged_steps += 1

        # Check whether a_unshielded would violate (always counted)
        would_violate = shield_controller is not None and self.would_violate(a_unshielded, context, shield_controller)

        # Count as a real violation if flags are active
        if flag_active and would_violate:
            self.episode_violations += 1
            self.total_violations += 1

        # Count modifications only if shield is active and actually changed action
        if flag_active and shield_controller is not None and shield_controller.is_shield_active:
            if a_unshielded != a_shielded:
                self.episode_modifications += 1
                self.total_modifications += 1

        if self.verbose and flag_active:
            print(f"[ConstraintMonitor] Flag: {flag_active}, Would Violate: {would_violate}, "
                  f"Modified: {a_unshielded != a_shielded}")

    def would_violate(self, action, context, shield_controller):
        """
        Returns True if the given action would be changed by the shield layer.
        """
        one_hot = torch.zeros(1, shield_controller.num_actions, dtype=torch.float32)
        one_hot[0, action] = 1.0

        # Get current flags
        flags = shield_controller.flag_logic_fn(context)
        flag_values = [flags.get(name, 0) for name in shield_controller.flag_names]
        flag_tensor = torch.tensor(flag_values, dtype=one_hot.dtype).unsqueeze(0)

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
