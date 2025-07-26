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

    def log_step(self, raw_probs, corrected_probs, selected_action, shield_controller, context, shield_applied=True):
        self.episode_steps += 1
        self.total_steps += 1

        # Get flags
        flags = shield_controller.flag_logic_fn(context)
        flag_values = [flags.get(name, 0) for name in shield_controller.flag_names]
        flag_active = any(flag_values)

        if flag_active:
            self.episode_flagged_steps += 1
            self.total_flagged_steps += 1

        # Compute unshielded action
        unshielded_action = raw_probs.argmax().item()
        modified = (unshielded_action != selected_action) if shield_applied else False
        would_violate = shield_controller.would_violate(unshielded_action, context)

        # === Only count violation if shield did not modify it ===
        count_violation = would_violate and not modified

        if not self.only_if_flags_active or flag_active:
            self.episode_modifications += modified
            self.total_modifications += modified

            self.episode_violations += count_violation
            self.total_violations += count_violation

        if self.verbose and flag_active:
            print(f"[ConstraintMonitor] Flags active. Mod: {modified}, Viol: {count_violation} (raw viol: {would_violate})")

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
