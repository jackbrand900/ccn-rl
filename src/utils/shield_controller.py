import itertools
from functools import partial

import torch
import os
import re
from pishield.propositional_requirements.shield_layer import ShieldLayer as PropositionalShieldLayer
from pishield.linear_requirements.shield_layer import ShieldLayer as LinearShieldLayer

from src.utils.env_helpers import is_in_front_of_key
from src.utils.req_file_to_logic_fn import get_flag_logic_fn

class ShieldController:
    def __init__(self, requirements_path, num_actions, mode="hard", verbose=False, default_flag_logic=None, is_shield_active=False):
        self.requirements_path = requirements_path
        self.num_actions = num_actions
        self.mode = mode
        self._base_flag_logic_fn = get_flag_logic_fn(self.requirements_path) or self.default_flag_logic
        if self.mode == "progressive":
            self.episode = 0
            flag_active_val = self.compute_progressive_flag(self.episode)
        else:
            flag_active_val = 0.8 if self.mode == "soft" else 1.0

        self.flag_logic_fn = partial(self._base_flag_logic_fn, flag_active_val=flag_active_val)
        self.flag_logic_batch = self._batchify(self.flag_logic_fn)

        # print(f"[DEBUG] Got flag logic function: {get_flag_logic_fn(self.requirements_path)}")

        # Parse var names from file
        self.var_names = self._extract_vars_from_requirements()
        self.num_vars = len(self.var_names)

        # Compute action and flag names based on position
        self.action_names = self.var_names[:num_actions]
        self.flag_names = self.var_names[num_actions:]
        self.num_flags = len(self.flag_names)
        self.verbose = verbose
        self.is_shield_active = is_shield_active

        self.ordering_names = [str(i) for i in range(self.num_vars)]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shield_layer = self.build_shield_layer().to(self.device)
        self.shield_activations = 0

        self.clauses = self._parse_cnf_file()
        self.sat_assignments_cache = {}

    def _batchify(self, single_fn):
        def batch_fn(contexts):
            return [single_fn(ctx) for ctx in contexts]
        return batch_fn

    def _extract_vars_from_requirements(self):
        with open(self.requirements_path, "r") as f:
            content = f.read()

        # Find all y_i style variables
        vars_found = set(re.findall(r"y_(\d+)", content))
        var_indices = sorted(int(v) for v in vars_found)

        if not var_indices:
            raise ValueError("No variables found in constraints file.")

        max_var = max(var_indices)
        all_vars = [f"y_{i}" for i in range(max_var + 1)]
        return all_vars

    def set_episode(self, episode):
        self.episode = episode
        if self.mode == "progressive":
            flag_active_val = self.compute_progressive_flag(episode)
            self.flag_logic_fn = partial(self._base_flag_logic_fn, flag_active_val=flag_active_val)
            self.flag_logic_batch = self._batchify(self.flag_logic_fn)

    def compute_progressive_flag(self, episode_num, final_episode=300) -> float:
        return min(1.0, 0.5 + 0.5 * (episode_num / final_episode))

    def flag_logic_with_key_check(self, context):
        """
        context: dict with keys 'obs' and 'direction', optionally 'position'
        Returns: dict like {'y_7': 1} if in front of key, assuming y_7 is your flag
        """
        obs = context['obs']  # shape (7,7,3) tensor or np.array
        direction = context['direction']  # 0=right, 1=down, 2=left, 3=up
        in_front = is_in_front_of_key(obs, direction)
        return {"y_7": int(in_front)}  # or whatever your flag var is

    def build_shield_layer(self):
        ordering = ",".join(reversed(self.ordering_names))  # e.g. "3,2,1,0"
        file_ext = os.path.splitext(self.requirements_path)[-1].lower()
        print(f"requirements file: {self.requirements_path}")
        print(f"num vars: {self.num_vars}, num flags: {self.num_flags}")
        print(f'Shield ordering: {ordering}')
        if file_ext == ".cnf":
            return PropositionalShieldLayer(
                num_classes=self.num_vars,
                requirements=self.requirements_path,
                ordering_choice="custom",
                custom_ordering=ordering,
            )
        elif file_ext == ".linear":
            return LinearShieldLayer(
                num_variables=self.num_vars,
                requirements_filepath=self.requirements_path,
                ordering_choice="given",
            )
        else:
            raise ValueError(f"Unknown requirements file extension: {file_ext}")

    def default_flag_logic(self, context):
        """Fallback flag logic — always 0 for all flags"""
        return {flag: 0 for flag in self.flag_names}

    def apply(self, action_probs, context):
        flags = self.flag_logic_fn(context)
        flag_values = [flags.get(name, 0) for name in self.flag_names]

        # Expand flags to match batch size
        batch_size = action_probs.size(0)  # batch dimension

        flag_tensor = torch.tensor(flag_values, dtype=action_probs.dtype, device=action_probs.device)

        # If flag_tensor is 1D, add batch dimension and expand to batch_size
        if flag_tensor.dim() == 1:
            flag_tensor = flag_tensor.unsqueeze(0).expand(batch_size, -1)

        # Check shapes for debug
        if action_probs.dim() != 2 or flag_tensor.dim() != 2:
            raise ValueError(f"Expected 2D tensors but got action_probs.dim()={action_probs.dim()}, flag_tensor.dim()={flag_tensor.dim()}")

        full_input = torch.cat([action_probs, flag_tensor], dim=1)
        shielded_output = self.shield_layer(full_input)
        corrected = shielded_output[:, :self.num_actions]
        if self.mode == "soft":
            corrected = corrected / corrected.sum(dim=1, keepdim=True)

        flag_active = any(flag_values)
        changed = not torch.allclose(action_probs, corrected, atol=1e-5)
        flags = self.flag_logic_fn(context)
        if self.verbose:
            print(f"[DEBUG] Raw flags: {flags}, Flag values: {flag_values}")
            if flag_active:
                print(f"[SHIELD ACTIVE] Flags: {flags}")
                if changed:
                    raw_np = action_probs.detach().cpu().numpy().flatten()
                    corrected_np = corrected.detach().cpu().numpy().flatten()
                    print(f"[SHIELD MODIFIED OUTPUT] Before: {raw_np} → After: {corrected_np}")
                else:
                    print(f"[SHIELD ACTIVE BUT NO CHANGE] Action output remained the same.")

        return corrected

    def apply_batch(self, action_probs, contexts):
        flag_dicts = self.flag_logic_batch(contexts)
        flag_values = [
            [flags.get(name, 0) for name in self.flag_names]
            for flags in flag_dicts
        ]
        flag_tensor = torch.tensor(flag_values, device=action_probs.device, dtype=action_probs.dtype)
        full_input = torch.cat([action_probs, flag_tensor], dim=1)
        shielded_output = self.shield_layer(full_input)
        corrected = shielded_output[:, :self.num_actions]

        if self.mode == "soft":
            corrected = corrected / corrected.sum(dim=1, keepdim=True)

        return corrected

    def forward_differentiable(self, action_probs, contexts):
        assert isinstance(contexts, list), "Contexts must be a list of dicts (batch)."
        flag_dicts = self.flag_logic_batch(contexts)
        flag_values = [
            [flags.get(name, 0.0) for name in self.flag_names]
            for flags in flag_dicts
        ]

        # Make sure flag_tensor is on the same device as action_probs
        flag_tensor = torch.tensor(
            flag_values, dtype=action_probs.dtype, device=action_probs.device
        )  # shape: [B, num_flags]

        full_input = torch.cat([action_probs, flag_tensor], dim=1)
        shielded_output = self.shield_layer(full_input)
        corrected = shielded_output[:, :self.num_actions]

        if self.mode == "soft":
            corrected = corrected / corrected.sum(dim=1, keepdim=True)

        # Count activations
        modified = ~torch.isclose(action_probs, corrected, atol=1e-5)
        activated = modified.any(dim=1)  # shape [B]
        self.shield_activations += activated.sum().item()

        if self.verbose:
            for i, (flags, raw, corr, act) in enumerate(zip(flag_dicts, action_probs, corrected, activated)):
                print(f"[DEBUG] Raw flags: {flags}")
                if any(flags.values()):
                    print(f"[SHIELD ACTIVE] Flags: {flags}")
                    if act:
                        raw_np = raw.detach().cpu().numpy().flatten()
                        corr_np = corr.detach().cpu().numpy().flatten()
                        print(f"[SHIELD MODIFIED OUTPUT] Before: {raw_np} → After: {corr_np}")
                    else:
                        print(f"[SHIELD ACTIVE BUT NO CHANGE] Action output remained the same.")

        return corrected

    def _parse_cnf_file(self):
        clauses = []
        with open(self.requirements_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                literals = re.split(r'\s+or\s+', line.strip())
                clause = []
                for lit in literals:
                    lit = lit.strip()
                    is_negated = lit.startswith('not ')
                    name = lit[4:] if is_negated else lit  # remove 'not '
                    if name not in self.var_names:
                        raise ValueError(f"Unknown variable '{name}' in CNF file.")
                    idx = self.var_names.index(name)
                    clause.append((idx, is_negated))
                clauses.append(clause)
        return clauses

    def _get_satisfying_assignments(self, flag_values):
        """
        For categorical actions: find which ACTION INDICES (0-17) satisfy constraints.
        Returns a list of action indices that satisfy the CNF when combined with flag_values.
        """
        key = tuple(flag_values)
        if key in self.sat_assignments_cache:
            return self.sat_assignments_cache[key]

        satisfying_actions = []

        # Check each possible action (0 to num_actions-1)
        for action_idx in range(self.num_actions):
            # Create one-hot encoding for this action
            action_bits = [0] * self.num_actions
            action_bits[action_idx] = 1

            # Combine with flag values
            full_bits = action_bits + list(flag_values)

            # Check if this satisfies all clauses
            if all(any(int(full_bits[idx]) ^ neg for idx, neg in clause) for clause in self.clauses):
                satisfying_actions.append(action_idx)

        # For categorical actions, we don't need sampling - just return all satisfying actions
        self.sat_assignments_cache[key] = satisfying_actions
        return satisfying_actions

    def compute_semantic_loss(self, action_probs, flag_tensor, debug=False):
        """
        Computes semantic loss for categorical actions (softmax distribution).
        action_probs: [B, num_actions] — softmax probabilities that sum to 1
        flag_tensor: [B, num_flags] — already-computed binary flags
        debug: if True, print detailed debugging info
        Returns: scalar semantic loss (averaged across batch)
        """
        device = action_probs.device
        B = action_probs.size(0)
        losses = []

        # Debug statistics
        total_satisfying_actions = 0
        total_possible_actions = 0
        min_satisfying_prob = float('inf')
        max_satisfying_prob = 0.0
        violation_count = 0

        for i in range(B):
            probs_i = action_probs[i]  # [num_actions]
            flag_values = flag_tensor[i].tolist()

            # Get satisfying action indices
            satisfying_actions = self._get_satisfying_assignments(flag_values)

            # Debug info for first few batches - only when constraints are restrictive
            is_restrictive = len(satisfying_actions) < self.num_actions
            if debug and i < 3 and is_restrictive:
                print(f"\n--- Batch item {i} (RESTRICTIVE CONSTRAINTS) ---")
                print(f"Flag values: {flag_values}")
                print(f"Satisfying actions: {satisfying_actions} ({len(satisfying_actions)}/{self.num_actions})")
                print(f"Action probs (top 5): {torch.topk(probs_i, 5)}")

            if not satisfying_actions:
                # No actions satisfy constraints - add large penalty
                violation_count += 1
                if debug:  # Always print violations
                    print(f"NO SATISFYING ACTIONS! Batch item {i}, Adding penalty of 10.0")
                losses.append(torch.tensor(10.0, device=device))
                continue

            # Convert to tensor of action indices
            sat_actions = torch.tensor(satisfying_actions, dtype=torch.long, device=device)

            # Get probabilities of satisfying actions
            satisfying_probs = probs_i[sat_actions]  # [num_satisfying_actions]

            # Total probability mass on satisfying actions
            total_satisfying_prob = satisfying_probs.sum()

            # Update debug statistics
            total_satisfying_actions += len(satisfying_actions)
            total_possible_actions += self.num_actions
            min_satisfying_prob = min(min_satisfying_prob, total_satisfying_prob.item())
            max_satisfying_prob = max(max_satisfying_prob, total_satisfying_prob.item())

            # Debug info for first few batches - only when constraints are restrictive
            if debug and i < 3 and is_restrictive:
                print(f"Satisfying probs: {satisfying_probs}")
                print(f"Total satisfying prob: {total_satisfying_prob.item():.8f}")
                print(f"Predicted action (argmax): {probs_i.argmax().item()}")
                print(f"Is predicted action valid: {probs_i.argmax().item() in satisfying_actions}")

            # Semantic loss is negative log of total probability on satisfying actions
            # Clamp to avoid numerical issues
            total_satisfying_prob = torch.clamp(total_satisfying_prob, min=1e-8, max=1.0)
            semantic_loss = -torch.log(total_satisfying_prob)
            losses.append(semantic_loss)

            if debug and i < 3 and is_restrictive:
                print(f"Individual semantic loss: {semantic_loss.item():.2e}")

        if not losses:
            return torch.tensor(0.0, device=device)

        # Return mean loss with numerical stability
        final_loss = torch.stack(losses).mean()

        # Print debug summary - only if there were restrictions or violations
        if debug:
            avg_satisfying_ratio = total_satisfying_actions / max(total_possible_actions, 1)
            restrictive_cases = total_satisfying_actions < total_possible_actions

            if restrictive_cases or violation_count > 0:
                print(f"\n=== SEMANTIC LOSS DEBUG SUMMARY ===")
                print(f"Batch size: {B}")
                print(f"Average satisfying actions ratio: {avg_satisfying_ratio:.3f}")
                print(f"Min satisfying prob: {min_satisfying_prob:.8f}")
                print(f"Max satisfying prob: {max_satisfying_prob:.8f}")
                print(f"Violations (no valid actions): {violation_count}/{B}")
                print(f"Final semantic loss: {final_loss.item():.2e}")
                print("=====================================\n")
            elif avg_satisfying_ratio == 1.0:
                print(f"All actions satisfy constraints for all {B} batch items (ratio: {avg_satisfying_ratio:.3f})")

        return torch.clamp(final_loss, min=0.0)