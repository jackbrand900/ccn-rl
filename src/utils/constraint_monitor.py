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

        probs_modified = not torch.allclose(raw_probs, corrected_probs, atol=epsilon)
        action_modified = a_unshielded != a_shielded
        violation = probs_modified and action_modified  # Heuristic violation definition

        # Check if any flags are active
        flag_active = True  # default to True if we don't require flags
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

            if shield_controller.is_shield_active:
                self.episode_modifications += int(action_modified)
                self.total_modifications += int(action_modified)

        if self.verbose and (not self.only_if_flags_active or flag_active):
            print(f"[ConstraintMonitor] Flags active: {flag_active}, Mod: {action_modified}, Viol: {violation}, Probs Modified: {probs_modified}")

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
