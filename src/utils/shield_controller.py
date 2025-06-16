import torch
from pishield.shield_layer import build_shield_layer

class ShieldController:
    def __init__(self, requirements_path, num_actions):
        self.requirements_path = requirements_path
        self.num_actions = num_actions
        self.var_names = self._parse_ordering()
        self.num_vars = len(self.var_names)
        self.num_flags = self._count_flags()
        self.shield_layer = self.build_shield_layer()

    def _parse_ordering(self):
        with open(self.requirements_path, 'r') as f:
            for line in f:
                if line.startswith("ordering"):
                    return line.strip().split()[1:]
        raise ValueError(f"No 'ordering' line found in {self.requirements_path}")

    def _count_flags(self):
        return len([v for v in self.var_names if int(v[2:]) >= self.num_actions])

    def build_shield_layer(self):
        self.shield_layer = build_shield_layer(
            self.num_vars,
            self.requirements_path,
            ordering_choice='given'
        )
        return self.shield_layer
