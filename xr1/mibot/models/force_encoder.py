"""Small force-only MLP and residual into XR-1's existing DiT AdaLN tensor."""

import torch
from torch import nn

from mibot.utils.force_contract import (
    FORCE_NORMALIZATION_ID, ForceEncoderConfig, WRENCH_START, WRENCH_STOP,
)


def split_force_state(state, normalization, *, dual_path=False):
    """Extract the current observation, without normalizing it a second time."""
    if not isinstance(state, torch.Tensor) or state.ndim != 3 or state.shape[1:] != (1, 60):
        raise ValueError("force encoder state must have shape (B,1,60), one current observation")
    if not state.is_floating_point():
        raise ValueError("force encoder state must be floating point")
    if not isinstance(normalization, torch.Tensor) or normalization.numel() != state.shape[0]:
        raise ValueError("force encoder requires one force_normalization marker per observation")
    if normalization.dtype != torch.int64 or not torch.all(normalization == FORCE_NORMALIZATION_ID):
        raise ValueError("force encoder requires state quantile normalization exactly once (marker 1)")
    wrench = state[..., WRENCH_START:WRENCH_STOP]
    if not torch.isfinite(wrench).all():
        raise ValueError("wrench contains nonfinite values")
    if torch.any(wrench.abs() > 1):
        raise ValueError("encoder expects quantile-normalized wrench in [-1,1], not raw N/Nm")
    if dual_path:
        return state, wrench
    proprio = state.clone()
    proprio[..., WRENCH_START:WRENCH_STOP] = 0
    return proprio, wrench


class ForceConditioner(nn.Module):
    def __init__(self, config=None, hidden_size=1024):
        super().__init__()
        self.config = ForceEncoderConfig.from_mapping(config)
        self.hidden_size = hidden_size
        # Isolate CPU initialization from the pretrained construction RNG stream,
        # including rank-dependent Lightning seeds. No CUDA RNG or device access.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.config.init_seed)
            dims = (6, *self.config.hidden_dims, self.config.output_dim)
            layers = []
            for index, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
                layer = nn.Linear(left, right, device="cpu", dtype=torch.float32)
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
                layers.append(layer)
                if index < len(dims) - 2:
                    layers.append(nn.SiLU())
            self.encoder = nn.Sequential(*layers)
            self.fusion = nn.Linear(self.config.output_dim, 6 * hidden_size, bias=False,
                                    device="cpu", dtype=torch.float32)
            if self.config.fusion_init == "zero":
                nn.init.zeros_(self.fusion.weight)
            else:
                nn.init.normal_(self.fusion.weight, std=self.config.fusion_init_std)
        self.encoder.requires_grad_(self.config.train_encoder)
        self.fusion.requires_grad_(self.config.train_fusion)

    def forward(self, wrench):
        if not isinstance(wrench, torch.Tensor) or wrench.ndim != 3 or wrench.shape[1:] != (1, 6) or not wrench.is_floating_point():
            raise ValueError("force encoder input must be floating (B,1,6)")
        if not torch.isfinite(wrench).all() or torch.any(wrench.abs() > 1):
            raise ValueError("force encoder input must be finite quantile-normalized values in [-1,1]")
        weight = self.encoder[0].weight
        # The original XR-1 forward uses bfloat16. Also support float32 CPU tests
        # and device/dtype conversion without copying data across devices silently.
        if wrench.device != weight.device:
            raise ValueError("wrench and force encoder must be on the same device")
        features = self.encoder(wrench.to(dtype=weight.dtype))
        return self.fusion(features).reshape(wrench.shape[0], 6, self.hidden_size)


def apply_force_train_policy(model, config):
    """Apply only explicit experiment-B policies; original mode is unchanged."""
    if not config.enabled:
        return
    if config.pretrained_policy != "original":
        model.requires_grad_(False)
        if config.pretrained_policy == "dit":
            for name in ("dit", "state_projector", "action_projector", "action_output_layer",
                         "t_embedder", "t_projector", "sink"):
                getattr(model, name).requires_grad_(True)
        elif config.pretrained_policy == "last_vlm_block":
            model.vlm.model.language_model.layers[-1].requires_grad_(True)
    model.force_conditioner.encoder.requires_grad_(config.train_encoder)
    model.force_conditioner.fusion.requires_grad_(config.train_fusion)
