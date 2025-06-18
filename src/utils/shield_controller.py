import torch
# from pishield.shield_layer import build_shield_layer
from pishield.propositional_requirements.shield_layer import ShieldLayer

class ShieldController:
    def __init__(self, requirements_path, num_actions, flag_logic_fn):
        self.requirements_path = requirements_path
        self.num_actions = num_actions
        self.var_names = ["y_{}".format(i) for i in range(8)] # TODO: hardcoded for now
        self.num_vars = len(self.var_names)
        self.num_flags = self._count_flags()
        self.flag_logic_fn = flag_logic_fn
        self.flag_names = self.var_names[self.num_actions:]
        self.shield_layer = self.build_shield_layer()


    def _count_flags(self):
        return len([v for v in self.var_names if int(v[2:]) >= self.num_actions])

    def build_shield_layer(self):
        shield = ShieldLayer(num_classes=8, # TODO: refactor this
                             requirements=self.requirements_path,
                             ordering_choice="custom",
                             custom_ordering="7,6,5,4,3,2,1,0")
        return shield

    def step_count_flag_logic(self, context):
        step = context["step"]
        flags = {"y_7": int(step % 10 == 0)}
        return flags

    def position_flag_logic(self, context):
        position = context["position"]
        flags = {"y_7": int(position == (1, 1))}
        return flags

    def apply(self, action_probs, context, verbose=False):
        flags = self.position_flag_logic(context)
        flag_values = [flags.get(name, 0) for name in self.flag_names]
        flag_tensor = torch.tensor(flag_values, device=action_probs.device, dtype=action_probs.dtype).unsqueeze(0)

        full_input = torch.cat([action_probs, flag_tensor], dim=1)
        shielded_output = self.shield_layer(full_input)
        corrected = shielded_output[:, :self.num_actions]

        # === Tracking ===
        flag_active = any(flag_values)
        changed = not torch.allclose(action_probs, corrected, atol=1e-5)
        if verbose:
            position = context["position"]
            print(f"Position: {position}")
            if flag_active:
                print(f"[SHIELD ACTIVE] Flags: {flags}")
                if changed:
                    print(f"[SHIELD MODIFIED OUTPUT] Before: {action_probs.cpu().numpy().flatten()} → After: {corrected.cpu().numpy().flatten()}")
                else:
                    print(f"[SHIELD ACTIVE BUT NO CHANGE] Action output remained the same.")

        return corrected
