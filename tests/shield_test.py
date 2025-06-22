import torch
import os
from src.utils.shield_controller import ShieldController
from pishield.propositional_requirements.shield_layer import ShieldLayer

def test_shield_controller():
    # Resolve constraint file path relative to project root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    constraint_path = os.path.join(BASE_DIR, "src", "requirements", "left_on_flag.cnf")
    num_actions = 7

    # --- Flag logic when y_7 is ON
    def flag_logic_flag_on(context):
        return {"y_7": 1}

    # --- Flag logic when y_7 is OFF
    def flag_logic_flag_off(context):
        return {"y_7": 0}

    # --- Fake action distribution (clearly violates left-only rule)
    action_probs = torch.tensor([[0.01, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165]], dtype=torch.float32)

    print("\n================ Test: Shield OFF ================\n")
    shield_off = ShieldController(
        requirements_path=constraint_path,
        num_actions=num_actions,
        flag_logic_fn=flag_logic_flag_off
    )
    out_off = shield_off.apply(action_probs.clone(), context={"step": 0})

    print(f"Original action_probs: {action_probs}")
    print(f"Output when flag OFF:  {out_off}")
    assert torch.allclose(action_probs, out_off, atol=1e-5), "[FAIL] Shield modified output when flag was off."

    print("\n================ Test: Shield ON ================\n")
    shield_on = ShieldController(
        requirements_path=constraint_path,
        num_actions=num_actions,
        flag_logic_fn=flag_logic_flag_on
    )
    out_on = shield_on.apply(action_probs.clone(), context={"step": 0})

    print(f"Original action_probs: {action_probs}")
    print(f"Output when flag ON:   {out_on}")

    # Detailed check
    y_0, rest = out_on[0, 0], out_on[0, 1:]
    print(f"y_0: {y_0.item():.4f}, other values: {rest.numpy()}")

    assert torch.isclose(y_0, torch.tensor(1.0), atol=1e-3), "[FAIL] y_0 not enforced to 1 when flag was on."
    assert torch.allclose(rest, torch.zeros_like(rest), atol=1e-3), "[FAIL] Other actions not forced to 0."

    print("\n[PASS] Shield correctly applied when flag was active.\n")

def test_logic_shield_enforcement():
    print("\n========== Test: Logic-Based Shield ==========")

    # Constraint file with logic: if y_7 then y_0=1, others=0
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    constraint_path = os.path.join(BASE_DIR, "src", "requirements", "left_on_flag.cnf")
    shield = ShieldLayer(num_classes=8,
                         requirements=constraint_path,
                         ordering_choice="custom",
                         custom_ordering="7,6,5,4,3,2,1,0")

    # --- Input: wrong distribution (violates the constraint if y_7=1)
    # Action probs: y_0=0.01, y_1..y_6=0.165, y_7=1 (flag active)
    action_probs = torch.tensor([[0.01, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 1.0]])

    # Apply logic shield
    output = shield(action_probs)

    # Extract corrected values
    y_corrected = output[0, :7]
    y7_out = output[0, 7]

    print("Original:", action_probs)
    print("Corrected:", output)
    print("y_0:", y_corrected[0].item(), "y_1–y_6:", y_corrected[1:].tolist(), "y_7:", y7_out.item())

    # Assertions
    assert y_corrected[0] > 0.9, "y_0 should be close to 1 when y_7 is active"
    assert all(y < 0.1 for y in y_corrected[1:]), "y_1–y_6 should be close to 0 when y_7 is active"
    assert y7_out > 0.9, "y_7 should remain active"

    print("[PASS] Logic-based constraint enforced correctly.")