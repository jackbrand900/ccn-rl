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

    def reset_all(self):
        self.reset()
        self.total_steps = 0
        self.total_modifications = 0
        self.total_violations = 0
        self.total_flagged_steps = 0

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

        # === Flag check ===
        flag_active = True
        if self.only_if_flags_active and context is not None and shield_controller is not None:
            flags = shield_controller.flag_logic_fn(context)
            # Get flags that are relevant and currently active
            active_flags = {name: val for name, val in flags.items()
                            if name in shield_controller.flag_names and val > 0}

            flag_active = len(active_flags) > 0
            if flag_active:
                self.episode_flagged_steps += 1
                self.total_flagged_steps += 1

        if self.verbose:
            print(f"[Monitor] Active flags: {list(active_flags.keys())}")

        if not flag_active:
            return  # no updates for unflagged steps

        # === Would the unshielded action have violated? ===
        violation = (
                shield_controller is not None and
                self.would_violate(a_unshielded, context, shield_controller)
        )

        # === Did the shield modify the action? ===
        modification = (a_unshielded != a_shielded) if shield_controller and shield_controller.is_shield_active else False

        # === Count violation if the action would have violated ===
        if violation:
            self.episode_violations += 1
            self.total_violations += 1

        # === Count modification only if shield was active, changed the action, and prevented a violation ===
        if shield_controller and shield_controller.is_shield_active and violation and modification:
            self.episode_modifications += 1
            self.total_modifications += 1

        if self.verbose:
            print(f"[ConstraintMonitor] Flags active: {flag_active}, "
                  f"Would Violate: {violation}, Modified: {modification}, "
                  f"Probs Modified: {not torch.allclose(raw_probs, corrected_probs, atol=epsilon)}")


    def would_violate(self, action, context, shield_controller, epsilon=1e-5):
        """
        Returns True if the given action would be blocked by the hard shield.

        This is used as a consistent way to log constraint violations regardless
        of whether the active shield mode is soft, post, or integrated.
        """
        device = next(shield_controller.shield_layer.parameters()).device

        # Simulate an action probability distribution (one-hot for the chosen action)
        one_hot_action_probs = torch.zeros(1, shield_controller.num_actions, dtype=torch.float32, device=device)
        one_hot_action_probs[0, action] = 1.0

        with torch.no_grad():
            # Let shield_controller handle flag construction and full input prep
            corrected = shield_controller.apply(one_hot_action_probs, context)
            corrected_probs = corrected[:, :shield_controller.num_actions]  # Ensure correct slice

        max_prob = corrected_probs.max().item()
        return corrected_probs[0, action].item() < (max_prob - epsilon)


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
