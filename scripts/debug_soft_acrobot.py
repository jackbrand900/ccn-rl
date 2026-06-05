"""Debug soft-mode shielding on Acrobot vs hard mode.

Question: is soft's poor tuning result a real mechanism or an implementation
artifact? We probe forward_differentiable for both modes on identical inputs.
"""
import torch
from src.utils.shield_controller import ShieldController

REQ = "src/requirements/acrobot_velocity_safe.cnf"
hard = ShieldController(REQ, num_actions=3, mode="hard", is_shield_active=True)
soft = ShieldController(REQ, num_actions=3, mode="soft", is_shield_active=True)

# Over-spin FORWARD => y_3 active => +torque (action 2) forbidden.
# threshold is 6.0 in raw rad/s, so use 7.0 to trigger.
ctx_on = {"theta2_dot": 7.0}
ctx_off = {"theta2_dot": 0.0}

print("=== flag value actually emitted ===")
print("hard flags on:", hard.flag_logic_fn(ctx_on))
print("soft flags on:", soft.flag_logic_fn(ctx_on))

def show(label, ctrl, probs, ctx):
    out = ctrl.forward_differentiable(probs.clone(), [ctx]).detach()
    print(f"{label:14s} in={probs.tolist()[0]}  ->  out={[round(x,3) for x in out.tolist()[0]]}  sum={out.sum():.3f}")
    return out

print("\n=== FLAG OFF (should be identity for both) ===")
p = torch.tensor([[0.2, 0.3, 0.5]])
show("hard off", hard, p, ctx_off)
show("soft off", soft, p, ctx_off)

print("\n=== FLAG ON, policy favors FORBIDDEN action 2 ===")
for probs in ([[0.1, 0.2, 0.7]], [[0.05, 0.05, 0.9]], [[0.33, 0.33, 0.34]]):
    p = torch.tensor(probs)
    show("hard on", hard, p, ctx_on)
    show("soft on", soft, p, ctx_on)
    print()

print("=== how much is the forbidden action (idx2) attenuated vs how much leaks ===")
p = torch.tensor([[0.1, 0.2, 0.7]])
h = hard.forward_differentiable(p.clone(), [ctx_on]).detach()
s = soft.forward_differentiable(p.clone(), [ctx_on]).detach()
print(f"raw P(forbidden)=0.700  | hard P(forbidden)={h[0,2]:.3f}  soft P(forbidden)={s[0,2]:.3f}")
print(f"  -> soft LEAKS {s[0,2]:.3f} prob mass onto the forbidden action (hard blocks fully)")

print("\n=== gradient flow check (soft) ===")
logits = torch.tensor([[0.1, 0.2, 0.7]], requires_grad=True)
probs = torch.softmax(logits, dim=-1)
out = soft.forward_differentiable(probs, [ctx_on])
loss = -torch.log(out[0, 0] + 1e-9)   # encourage safe action 0
loss.backward()
print("logits.grad:", logits.grad.tolist()[0], "finite?", bool(torch.isfinite(logits.grad).all()))
