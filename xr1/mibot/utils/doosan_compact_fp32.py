"""Experimental XR1 last-N FP32 trainables + compact Lightning full-state checkpoints.

Install *after* original BaseRunner.configure_model() has loaded the pretrained
baseline and after the exact last-N policy has been applied. Before DDP wrapping
and optimizer construction. Intended for bf16-mixed autocast and torch AdamW.

Never invoke on a loaded full-checkpoint resume; compact format only. The final
checkpoint must be accompanied by the exact immutable pretrained checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import types
from collections.abc import Iterable
from typing import Any

import torch

FORMAT = "xr1-doosan-fp32-trainable-compact-lightning-v1"


def _keyhash(keys: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()


def install_compact_fp32(
    runner: torch.nn.Module,
    *,
    selected_keys: Iterable[str],
    pretrained_sha256: str,
    language_sha256: str,
    source_commit: str,
    last_n: int,
) -> dict[str, Any]:
    """Install strict partial-state checkpointing and promote trainables to FP32.

    `runner` must have an initialized `.model` and should be a LightningModule.
    `runner.model` is expected to have the audited requires_grad freeze policy.
    """
    if getattr(runner, "_xr1_compact_fp32_installed", False):
        raise RuntimeError("Compact FP32 hook already installed")
    if not hasattr(runner, "strict_loading"):
        raise TypeError("Expected a LightningModule with strict_loading")
    selected = set(selected_keys)
    named = dict(runner.named_parameters())
    actual = {name for name, p in named.items() if p.requires_grad}
    if actual != selected or not selected:
        raise RuntimeError(f"Freeze policy mismatch: expected={len(selected)} actual={len(actual)} "
                           f"missing={sorted(selected - actual)[:5]} unexpected={sorted(actual - selected)[:5]}")
    if not isinstance(last_n, int) or last_n < 1:
        raise ValueError("last_n must be positive")
    if not all(len(value) == 64 and all(c in "0123456789abcdef" for c in value)
               for value in (pretrained_sha256, language_sha256)):
        raise ValueError("Expected lowercase SHA256 for pretrained and language policy")
    if not source_commit or not isinstance(source_commit, str):
        raise ValueError("source_commit is required")
    full_state_keys = set(runner.state_dict())
    if not selected.issubset(full_state_keys):
        raise RuntimeError("Trainable parameter keys absent from full state_dict")

    # The FP32 trainable parameter is itself the master weight. bf16-mixed
    # autocast performs bf16 compute without rounding the stored FP32 master.
    total = 0
    for key in sorted(selected):
        p = named[key]
        if p.device.type == "meta":
            raise RuntimeError("Cannot promote meta/sharded parameter on this path")
        if p.dtype not in (torch.bfloat16, torch.float32):
            raise RuntimeError(f"Unexpected selected parameter dtype: {key}={p.dtype}")
        with torch.no_grad():
            p.data = p.detach().to(dtype=torch.float32).contiguous()
        total += p.numel()
        if not p.requires_grad or p.dtype != torch.float32:
            raise RuntimeError(f"Failed to promote {key} to FP32")

    metadata = {
        "schema": FORMAT,
        "selected_key_sha256": _keyhash(selected),
        "selected_tensor_count": len(selected),
        "trainable_parameter_count": total,
        "pretrained_sha256": pretrained_sha256,
        "language_sha256": language_sha256,
        "source_commit": source_commit,
        "last_n": last_n,
        "trainable_storage_dtype": "torch.float32",
        "optimizer": "torch.optim.AdamW with fp32 exp_avg and exp_avg_sq",
        "frozen_weights_source": "external_original_pretrained_checkpoint",
    }
    # Lightning supports partial-model checkpointing when strict_loading=False,
    # but check missing keys EXACTLY on restore instead of ignoring surprises.
    raw_state_dict = runner.state_dict
    raw_load_state_dict = runner.load_state_dict
    original_save = runner.on_save_checkpoint
    original_load = runner.on_load_checkpoint
    runner.strict_loading = False

    def compact_state_dict(self, *args, **kwargs):
        full = raw_state_dict(*args, **kwargs)
        if set(full) != full_state_keys:
            raise RuntimeError("Unexpected model state_dict key drift")
        result = {name: full[name] for name in sorted(selected)}
        for key, value in result.items():
            if value.dtype != torch.float32:
                raise RuntimeError(f"Selected model state not FP32: {key}")
        return result

    def compact_load_state_dict(self, state_dict, strict=True, **kwargs):
        if set(state_dict) != selected:
            raise RuntimeError("Partial model checkpoint key mismatch: "
                               f"missing={sorted(selected-set(state_dict))[:6]} "
                               f"extra={sorted(set(state_dict)-selected)[:6]}")
        for name, value in state_dict.items():
            if value.dtype != torch.float32 or value.shape != named[name].shape:
                raise RuntimeError(f"Bad master weight dtype/shape: {name}")
        result = raw_load_state_dict(state_dict, strict=False, **kwargs)
        if set(result.missing_keys) != full_state_keys - selected or result.unexpected_keys:
            raise RuntimeError("Unexpected frozen/extra keys during partial checkpoint load")
        return result

    def compact_save_checkpoint(self, checkpoint):
        original_save(checkpoint)
        if set(checkpoint["state_dict"]) != selected:
            raise RuntimeError("Compact checkpoint unexpectedly contains frozen weights")
        if not checkpoint.get("optimizer_states"):
            raise RuntimeError("Optimizer state missing from compact resume checkpoint")
        for optimizer in checkpoint["optimizer_states"]:
            states = optimizer.get("state", {})
            for state in states.values():
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state and state[key].dtype != torch.float32:
                        raise RuntimeError(f"Optimizer moment not FP32: {key}")
        checkpoint["xr1_compact_fp32"] = dict(metadata)

    def compact_load_checkpoint(self, checkpoint):
        actual_info = checkpoint.get("xr1_compact_fp32")
        if actual_info != metadata:
            raise RuntimeError("Compact metadata mismatch (base/checksum/freeze/format)")
        if set(checkpoint.get("state_dict", {})) != selected:
            raise RuntimeError("Compact checkpoint selected parameter keys differ")
        original_load(checkpoint)

    runner.state_dict = types.MethodType(compact_state_dict, runner)
    runner.load_state_dict = types.MethodType(compact_load_state_dict, runner)
    runner.on_save_checkpoint = types.MethodType(compact_save_checkpoint, runner)
    runner.on_load_checkpoint = types.MethodType(compact_load_checkpoint, runner)
    runner._xr1_compact_fp32_installed = True
    runner._xr1_compact_fp32_metadata = metadata
    return dict(metadata)
