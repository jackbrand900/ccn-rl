"""Reproduce the hard-layer crash on Acrobot end-to-end through the REAL shield
code path. Mirrors network.PPONetwork.select_action exactly:
    pre_probs = softmax(logits)
    post_probs = shield_controller.forward_differentiable(pre_probs, context)
    action = Categorical(probs=post_probs).sample()
We craft logits peaked on the FORBIDDEN action while the over-spin flag is
active, so the shield zeros it and the legal actions underflow to 0.0.
"""
import torch
from torch.distributions import Categorical
from src.utils.shield_controller import ShieldController

REQ = "src/requirements/acrobot_velocity_safe.cnf"
ctrl = ShieldController(REQ, num_actions=3, mode="hard", verbose=False, is_shield_active=True)

# Over-spin FORWARD => flag y_3 active => +torque (action 2) is forbidden.
context = {"theta2_dot": 0.5}

for gap in [40, 90, 104, 200]:
    # Policy is extremely confident on the FORBIDDEN action (index 2).
    logits = torch.tensor([[0.0, 0.0, float(gap)]])
    pre_probs = torch.softmax(logits, dim=-1)
    post_probs = ctrl.forward_differentiable(pre_probs, [context])
    print(f"\ngap={gap}: pre={pre_probs.tolist()[0]}  post={post_probs.tolist()[0]}  sum={post_probs.sum().item():.3e}")
    try:
        dist = Categorical(probs=post_probs)
        a = dist.sample().item()
        print(f"  sampled action={a} (no crash)")
    except Exception as e:
        print(f"  *** CRASH: {type(e).__name__}: {str(e)[:120]}")

    # Deterministic (eval) path: argmax + log_prob
    a_det = post_probs.argmax(dim=-1)
    lp = torch.log(post_probs.gather(1, a_det.unsqueeze(1))).squeeze(1).item()
    forbidden = (a_det.item() == 2)
    print(f"  deterministic: action={a_det.item()}{'  <-- FORBIDDEN +torque!' if forbidden else ''}  log_prob={lp}")
