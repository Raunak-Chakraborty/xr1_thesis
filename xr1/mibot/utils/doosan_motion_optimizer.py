"""Opt-in optimizer routing for XR-1 motion-focused training.

This does not modify production AdamW or checkpoint handling.
Muon backend: pytorch-optimizer 4.1.0, when installed.
"""
from __future__ import annotations

import hashlib
import json

import torch


BACKEND = "pytorch-optimizer==4.1.0"

LAST_ONE_HIDDEN = (
    ".visual.blocks.23.",
    ".language_model.layers.35.",
    ".dit.layers.35.",
)

_NO_DECAY = (
    "bias",
    "norm",
    "ln",
    "rotary_emb",
    "adaln",
)


def route_last_one(named_parameters):
    """Partition trainable FP32 parameters without duplication."""

    groups = {
        "muon_hidden_matrix": [],
        "adamw_decay": [],
        "adamw_no_decay": [],
    }

    receipt = []
    seen_ids = set()
    names = set()

    for name, param in named_parameters:
        if not param.requires_grad:
            continue

        if name in names or id(param) in seen_ids:
            raise ValueError(
                "Duplicate trainable name/tensor: " + name
            )

        names.add(name)
        seen_ids.add(id(param))

        if param.dtype != torch.float32:
            raise ValueError(
                f"Non-FP32 trainable weight: {name} {param.dtype}"
            )

        if (
            param.ndim == 2
            and any(
                marker in name
                for marker in LAST_ONE_HIDDEN
            )
        ):
            group = "muon_hidden_matrix"

        elif (
            param.ndim < 2
            or any(
                marker in name.lower()
                for marker in _NO_DECAY
            )
        ):
            group = "adamw_no_decay"

        else:
            group = "adamw_decay"

        groups[group].append(param)

        receipt.append((
            name,
            group,
            list(param.shape),
        ))

    if not receipt or not groups["muon_hidden_matrix"]:
        raise ValueError(
            "Last-one routing found no Muon-eligible hidden matrices"
        )

    canonical = json.dumps(
        sorted(receipt),
        separators=(",", ":")
    )

    report = {
        "schema": "xr1-motion-hybrid-routing-v1",
        "sha256": hashlib.sha256(
            canonical.encode()
        ).hexdigest(),

        "tensor_counts": {
            key: len(values)
            for key, values in groups.items()
        },

        "element_counts": {
            key: sum(
                param.numel()
                for param in values
            )
            for key, values in groups.items()
        },

        "total_trainable_tensors": len(receipt),

        "total_trainable_elements": sum(
            param.numel()
            for values in groups.values()
            for param in values
        ),

        "backend": BACKEND,
    }

    return groups, report


def create_motion_optimizer(
    named_parameters,
    *,
    kind,
    adamw_lr,
    muon_lr=None,
    adamw_weight_decay=0.01,
    muon_weight_decay=0.01,
    muon_momentum=0.95,
):
    """Construct one optimizer for Lightning-compatible scheduling.

    Supported:
        adamw_control: regular torch AdamW
        hybrid_muon: pinned pytorch-optimizer Muon
                     with AdamW fallback groups

    Muon never silently falls back when its backend is absent.
    """

    if kind not in ("adamw_control", "hybrid_muon"):
        raise ValueError(
            "Expected adamw_control or hybrid_muon"
        )

    if not 0 < float(adamw_lr) < 1:
        raise ValueError("Invalid AdamW learning rate")

    if kind == "hybrid_muon":
        if muon_lr is None or not 0 < float(muon_lr) < 1:
            raise ValueError("Muon learning rate required")

    groups, report = route_last_one(
        named_parameters
    )

    configs = []

    for name in (
        "muon_hidden_matrix",
        "adamw_decay",
        "adamw_no_decay",
    ):
        if not groups[name]:
            continue

        uses_muon = (
            kind == "hybrid_muon"
            and name == "muon_hidden_matrix"
        )

        lr = (
            muon_lr if uses_muon
            else adamw_lr
        )

        if uses_muon:
            wd = muon_weight_decay
        elif name == "adamw_no_decay":
            wd = 0.0
        else:
            wd = adamw_weight_decay

        configs.append({
            "params": groups[name],
            "group_name": name,
            "use_muon": uses_muon,
            "lr": float(lr),
            "weight_decay": float(wd),
        })

    if kind == "adamw_control":
        optimizer = torch.optim.AdamW(
            configs,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    else:
        import importlib.metadata

        try:
            version = importlib.metadata.version(
                "pytorch-optimizer"
            )
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "Muon requires pinned "
                "pytorch-optimizer==4.1.0. "
                "Install in the qualified experimental runtime."
            ) from exc

        if version != "4.1.0":
            raise RuntimeError(
                "Muon backend version mismatch: "
                + version
            )

        from pytorch_optimizer import Muon

        optimizer = Muon(
            configs,
            lr=float(muon_lr),
            momentum=float(muon_momentum),
            weight_decay=float(muon_weight_decay),
            nesterov=True,
            ns_steps=5,
        )

    report = {
        **report,
        "optimizer_kind": kind,
        "param_group_count": len(configs),
    }

    return optimizer, report
