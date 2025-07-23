import torch
from pishield.propositional_requirements.shield_layer import ShieldLayer

# Format: y_0, y_1 = actions; y_2, y_3 = flags
x = torch.tensor([[0.3, 0.7, 0.0, 1.0]])  # y_3 is active → enforce y_1, disable y_0

sl = ShieldLayer(
    num_classes=4,
    requirements="src/requirements/emergency_cartpole.cnf",
    ordering_choice="custom",
    custom_ordering="3,2,1,0"  # y_3, y_2, y_1, y_0
)

with torch.no_grad():
    out = sl(x)

print("Input:           ", x.numpy())
print("Shielded output: ", out.numpy())
