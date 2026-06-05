"""Quick functional probe: does the CCN+ (PiShield) layer enforce the
Acrobot joint-velocity constraint correctly? No training — just build the
shield on acrobot_velocity_safe.cnf and feed it one-hot actions under each
flag condition."""
import torch
from src.utils.shield_controller import ShieldController

REQ = "src/requirements/acrobot_velocity_safe.cnf"
# Acrobot: 3 actions (0=-torque, 1=0 torque, 2=+torque)
ctrl = ShieldController(REQ, num_actions=3, mode="hard", verbose=False, is_shield_active=True)

print(f"var_names={ctrl.var_names}  actions={ctrl.action_names}  flags={ctrl.flag_names}")
print(f"clauses={ctrl.clauses}")


def probe(label, action, context):
    one_hot = torch.zeros(1, 3)
    one_hot[0, action] = 1.0
    out = ctrl.apply(one_hot, context).detach().cpu().numpy().flatten()
    kept = out[action]
    verdict = "BLOCKED" if kept < 0.999 else "allowed"
    print(f"{label:42s} action={action} -> probs={out.round(3)}  [{verdict}]")


# theta2_dot normalized; threshold 0.21. y_3 fires when >0.21, y_4 when <-0.21.
overspin_fwd = {"theta2_dot": 0.5}    # y_3 active
overspin_bwd = {"theta2_dot": -0.5}   # y_4 active
no_flag = {"theta2_dot": 0.0}         # neither

print("\n-- overspin FORWARD (y_3): +torque (action 2) should be blocked --")
probe("overspin_fwd + apply +torque", 2, overspin_fwd)
probe("overspin_fwd + apply -torque (opposite, ok)", 0, overspin_fwd)
probe("overspin_fwd + apply 0 torque (ok)", 1, overspin_fwd)

print("\n-- overspin BACKWARD (y_4): -torque (action 0) should be blocked --")
probe("overspin_bwd + apply -torque", 0, overspin_bwd)
probe("overspin_bwd + apply +torque (opposite, ok)", 2, overspin_bwd)

print("\n-- no flag active: nothing should change --")
probe("no_flag + apply +torque", 2, no_flag)
probe("no_flag + apply -torque", 0, no_flag)
