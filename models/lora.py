"""
Token-Level LoRA for UniMotion (Show-o2 with motion modality).
Implements token-type-based dual-path LoRA for the LLM's attention layers.

Paper: Modality-Routed LoRA (LoRA-A: text/RGB, LoRA-B: motion)

Core idea:
- LoRA-A: for Text/Image tokens (modality_mask == 0)
- LoRA-B: for Motion tokens (modality_mask == 1)
- The modality_mask selects which LoRA to use at the token level

Q/K/V/O computation flow:
    Q_base = W_q @ hidden_states
    Q_lora_A = ΔW_q_A @ hidden_states  (Text/Image)
    Q_lora_B = ΔW_q_B @ hidden_states  (Motion)

    mask = modality_mask.unsqueeze(-1)
    Q = Q_base + Q_lora_A * (1-mask) + Q_lora_B * mask
"""

import math
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional


class LoRALayer(nn.Module):
    """
    A single LoRA adapter.

    Implements the low-rank decomposition ΔW = B @ A, where:
    - A: [rank, in_features] down-projection matrix
    - B: [out_features, rank] up-projection matrix
    - scaling = alpha / rank controls the strength of the LoRA contribution
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.in_features = in_features
        self.out_features = out_features

        # LoRA matrices: ΔW = B @ A
        # A: down-projection [rank, in_features]
        # B: up-projection [out_features, rank]
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout (optional)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialization: A uses Kaiming uniform init, B is zero-initialized
        # so that ΔW = 0 initially and the original model behavior is unchanged
        self._init_weights()

    def _init_weights(self):
        """Initialize LoRA weights."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the LoRA delta output.

        Args:
            x: [bs, seq_len, in_features] input tensor

        Returns:
            [bs, seq_len, out_features] LoRA delta output (ΔW @ x * scaling)
        """
        # x @ A.T @ B.T * scaling
        # [bs, seq_len, in_features] @ [in_features, rank] -> [bs, seq_len, rank]
        # [bs, seq_len, rank] @ [rank, out_features] -> [bs, seq_len, out_features]
        return (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling


class TokenWiseLoRALinear(nn.Module):
    """
    Token-Level dual-path LoRA Linear layer.

    Paper: Modality-Routed LoRA (LoRA-A: text/RGB, LoRA-B: motion)

    Selects LoRA-A (text/image) or LoRA-B (motion) based on modality_mask.
    This allows different token types within the same sequence to use
    different LoRA adapters.

    Usage:
    - modality_mask == 0: Text/Image tokens -> use LoRA-A
    - modality_mask == 1: Motion tokens -> use LoRA-B

    Computation:
        output = base_linear(x) + lora_A(x) * (1-mask) + lora_B(x) * mask
    """

    # Marker attribute used at forward time to decide whether modality_mask
    # needs to be passed in
    is_token_wise_lora = True

    def __init__(
        self,
        base_linear: nn.Linear,
        rank_A: int = 16,
        rank_B: int = 16,
        alpha: float = 16,
        dropout: float = 0.0,
        enable_A: bool = True,
        enable_B: bool = True,
    ):
        """
        Initialize TokenWiseLoRALinear.

        Args:
            base_linear: original nn.Linear layer (will be frozen)
            rank_A: rank of LoRA-A (Text/Image path)
            rank_B: rank of LoRA-B (Motion path)
            alpha: LoRA alpha parameter (scaling = alpha/rank)
            dropout: LoRA dropout rate
            enable_A: whether to enable LoRA-A (Text/Image path)
            enable_B: whether to enable LoRA-B (Motion path)
        """
        super().__init__()

        # Keep the original Linear layer
        self.base_linear = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        # Get the device and dtype of base_linear so LoRA layers live on the same device
        device = base_linear.weight.device
        dtype = base_linear.weight.dtype

        # Freeze base weights - excluded from gradient updates
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        # LoRA-A: Text/Image path
        self.enable_A = enable_A
        if enable_A:
            self.lora_A = LoRALayer(
                self.in_features,
                self.out_features,
                rank_A,
                alpha,
                dropout
            ).to(device=device, dtype=dtype)
        else:
            self.lora_A = None

        # LoRA-B: Motion path
        self.enable_B = enable_B
        if enable_B:
            self.lora_B = LoRALayer(
                self.in_features,
                self.out_features,
                rank_B,
                alpha,
                dropout
            ).to(device=device, dtype=dtype)
        else:
            self.lora_B = None

    def forward(
        self,
        x: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass; selects the LoRA path based on modality_mask.

        Args:
            x: [bs, seq_len, in_features] input tensor
            modality_mask: [bs, seq_len] modality mask
                - 0: text/image tokens (use LoRA-A)
                - 1: motion tokens (use LoRA-B)
                - None: use base only (no LoRA)

        Returns:
            [bs, seq_len, out_features] output tensor
        """
        # Base output (always computed)
        output = self.base_linear(x)

        # If there is no mask, return the base output only
        if modality_mask is None:
            return output

        # Expand mask for broadcasting: [bs, seq_len] -> [bs, seq_len, 1]
        mask = modality_mask.unsqueeze(-1).to(x.dtype)

        # Apply LoRA-A (text/image tokens where mask == 0)
        if self.enable_A and self.lora_A is not None:
            lora_A_out = self.lora_A(x)
            output = output + lora_A_out * (1 - mask)

        # Apply LoRA-B (motion tokens where mask == 1)
        if self.enable_B and self.lora_B is not None:
            lora_B_out = self.lora_B(x)
            output = output + lora_B_out * mask

        return output

    @property
    def weight(self) -> torch.Tensor:
        """
        Compatibility property: returns the weight of base_linear.
        Some code (e.g. FlashAttention2's dtype detection) accesses the .weight attribute.
        """
        return self.base_linear.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Compatibility property: returns the bias of base_linear."""
        return self.base_linear.bias

    def extra_repr(self) -> str:
        """Return the module's extra representation string."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"enable_A={self.enable_A}, "
            f"enable_B={self.enable_B}"
        )


def apply_token_wise_lora(
    model: nn.Module,
    config: Dict,
) -> None:
    """
    Apply Token-Level LoRA to the model's attention layers.

    Paper: Modality-Routed LoRA (LoRA-A: text/RGB, LoRA-B: motion)

    Iterates over all self_attn modules in the model and replaces the target
    Linear layers with TokenWiseLoRALinear.

    Args:
        model: Qwen2ForCausalLM model (or a wrapper containing it)
        config: LoRA config dict with the following keys:
            - rank_A: rank of LoRA-A (default 16)
            - rank_B: rank of LoRA-B (default 16)
            - alpha: LoRA alpha (default 16)
            - dropout: LoRA dropout (default 0.0)
            - enable_A: whether to enable LoRA-A (default True)
            - enable_B: whether to enable LoRA-B (default True)
            - target_modules: list of modules to apply LoRA to (default ['q_proj', 'k_proj', 'v_proj', 'o_proj'])
    """
    rank_A = config.get('rank_A', 16)
    rank_B = config.get('rank_B', 16)
    alpha = config.get('alpha', 16)
    dropout = config.get('dropout', 0.0)
    enable_A = config.get('enable_A', True)
    enable_B = config.get('enable_B', True)
    target_modules = config.get('target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj'])

    # Collect all modules that need to be replaced
    modules_to_replace = []
    already_applied_count = 0

    # Iterate over all modules
    for name, module in model.named_modules():
        # Check whether this is a target module (q/k/v/o_proj inside self_attn)
        for target in target_modules:
            if name.endswith(f'.self_attn.{target}'):
                # Guard against repeated calls: skip if it is already a TokenWiseLoRALinear
                if isinstance(module, TokenWiseLoRALinear):
                    already_applied_count += 1
                    continue
                # Make sure it is an nn.Linear
                if not isinstance(module, nn.Linear):
                    print(f"[LoRA] Warning: {name} is not nn.Linear, skipping")
                    continue
                modules_to_replace.append((name, target, module))

    if already_applied_count > 0:
        print(f"[LoRA] Warning: {already_applied_count} modules already have LoRA applied, skipping them")

    # Perform the replacement
    for full_name, target_name, old_linear in modules_to_replace:
        # Get the parent module (self_attn)
        parent_name = full_name.rsplit('.', 1)[0]
        parent = model.get_submodule(parent_name)

        # Create a TokenWiseLoRALinear to replace the original Linear
        new_linear = TokenWiseLoRALinear(
            old_linear,
            rank_A=rank_A,
            rank_B=rank_B,
            alpha=alpha,
            dropout=dropout,
            enable_A=enable_A,
            enable_B=enable_B,
        )

        # Replace the module
        setattr(parent, target_name, new_linear)

    print(f"[LoRA] Applied Token-Wise LoRA to {len(modules_to_replace)} modules")
    print(f"[LoRA] Config: rank_A={rank_A}, rank_B={rank_B}, alpha={alpha}, "
          f"enable_A={enable_A}, enable_B={enable_B}")


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract all LoRA parameters from the model.

    Args:
        model: model containing LoRA layers

    Returns:
        state_dict containing all LoRA parameters (CPU tensors, for easy cross-device loading)
    """
    state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            # Save as CPU tensors to avoid cross-device loading issues
            state_dict[name] = param.data.clone().cpu()
    return state_dict


def load_lora_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load LoRA parameters into the model.

    Args:
        model: target model
        state_dict: LoRA parameter dict
        strict: whether to require strict parameter-name matching
    """
    loaded_keys = []
    missing_keys_in_model = []
    missing_keys_in_state = []

    model_params = dict(model.named_parameters())

    # Use no_grad to avoid autograd tracking
    with torch.no_grad():
        for name, tensor in state_dict.items():
            if name in model_params:
                param = model_params[name]
                # Move the tensor to the target device and dtype before copying
                param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
                loaded_keys.append(name)
            else:
                missing_keys_in_model.append(name)

    loaded_key_set = set(loaded_keys)
    model_lora_keys = [name for name in model_params.keys() if 'lora_' in name]
    for name in model_lora_keys:
        if name not in loaded_key_set:
            missing_keys_in_state.append(name)

    if missing_keys_in_model and strict:
        raise RuntimeError(f"Keys in state_dict not found in model: {missing_keys_in_model}")

    # For callers saving across stages: keep LoRA branch parameters that are
    # not instantiated in the current model
    unmatched_state_dict = {
        name: state_dict[name].clone().cpu()
        for name in missing_keys_in_model
    }

    print(f"[LoRA] Loaded {len(loaded_keys)} LoRA parameters")
    if missing_keys_in_model:
        print(
            f"[LoRA] Warning: {len(missing_keys_in_model)} keys from checkpoint "
            f"not found in current model (strict={strict})"
        )
    if missing_keys_in_state:
        print(
            f"[LoRA] Info: {len(missing_keys_in_state)} current-model LoRA keys "
            f"were not found in checkpoint"
        )

    return {
        'loaded_keys': loaded_keys,
        'missing_keys_in_model': missing_keys_in_model,
        'missing_keys_in_state': missing_keys_in_state,
        'unmatched_state_dict': unmatched_state_dict,
    }


def count_lora_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count the number of LoRA parameters in the model.

    Note the parameter naming convention:
    - `...q_proj.lora_A.lora_A` - A matrix of the Text/Image path (lora_A module)
    - `...q_proj.lora_A.lora_B` - B matrix of the Text/Image path (lora_A module)
    - `...q_proj.lora_B.lora_A` - A matrix of the Motion path (lora_B module)
    - `...q_proj.lora_B.lora_B` - B matrix of the Motion path (lora_B module)

    The counts here are grouped by **path** (Text/Image vs Motion), not by matrix type.
    The path module is identified from the **second-to-last** component of the parameter name.

    Args:
        model: model containing LoRA layers

    Returns:
        dict with parameter counts:
        - text_image_path_params: parameter count of the Text/Image path (LoRA-A module)
        - motion_path_params: parameter count of the Motion path (LoRA-B module)
        - total_lora_params: total LoRA parameter count
    """
    text_image_path_params = 0  # lora_A module (Text/Image path)
    motion_path_params = 0      # lora_B module (Motion path)
    total_lora_params = 0

    for name, param in model.named_parameters():
        if 'lora_' in name:
            num_params = param.numel()
            total_lora_params += num_params

            # Identify the LoRA path module from the second-to-last component of the name
            # Name format: ...q_proj.lora_A.lora_A or ...q_proj.lora_B.lora_B
            # The second-to-last component (lora_A or lora_B) denotes the path module
            parts = name.split('.')
            if len(parts) >= 2:
                # The second-to-last component is the module name (lora_A or lora_B)
                module_name = parts[-2]
                if module_name == 'lora_A':
                    text_image_path_params += num_params
                elif module_name == 'lora_B':
                    motion_path_params += num_params

    return {
        'text_image_path_params': text_image_path_params,
        'motion_path_params': motion_path_params,
        'total_lora_params': total_lora_params,
    }


def freeze_non_lora_params(model: nn.Module) -> None:
    """
    Freeze all non-LoRA parameters.

    Args:
        model: target model
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False
            frozen_count += 1

    print(f"[LoRA] Froze {frozen_count} non-LoRA parameters")


def unfreeze_lora_params(model: nn.Module) -> None:
    """
    Unfreeze all LoRA parameters (ensure they are trainable).

    Args:
        model: target model
    """
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
            unfrozen_count += 1

    print(f"[LoRA] Unfroze {unfrozen_count} LoRA parameters")
