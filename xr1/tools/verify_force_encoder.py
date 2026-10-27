#!/usr/bin/env python3
"""Offline workstation-only real XR1 checkpoint/gradient verification.

Requires the existing XR-1 environment and cached Qwen config/processor. It
never starts a server or robot client, downloads assets, or installs packages.
Synthetic statistics here are solely for this synthetic verification.
"""

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Existing consolidated XR-1 weights (plain, module, or state_dict)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory; full model plus optimizer checkpoint needs disk space")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-repeat", type=int, default=1,
                        help="Memory-conscious verification; production default is 4")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit("--checkpoint must be an existing consolidated checkpoint file")
    if args.output_dir.exists():
        raise SystemExit("Refusing to overwrite --output-dir; choose a fresh directory")
    if not args.device.startswith("cuda") or args.training_repeat < 1:
        raise SystemExit("This is a workstation CUDA check; training-repeat must be positive")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    import torch
    from PIL import Image
    from mibot.models.VLA.XR1 import xr1
    from mibot.data.collate.custom_collate import CustomCollate
    from mibot.utils.force_checkpoint import load_force_checkpoint, model_weights
    from mibot.utils.force_normalization import normalize_force_state

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; no GPU verification executed")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "synthetic_only": True, "checkpoint": str(args.checkpoint),
              "torch_version": torch.__version__, "device": torch.cuda.get_device_name(device)}
    try:
        config = dict(enabled=True, pretrained_policy="frozen", init_seed=args.seed)
        model = xr1(force_encoder=config, async_train=False, freq_excluded_dims=[6])
        checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
        load_force_checkpoint(model, model_weights(checkpoint), allow_base_checkpoint=True)
        del checkpoint
        model.to(device)
        model.training_repeat = args.training_repeat
        report["force_config"] = asdict(model.force_encoder_config)
        report["parameters"] = {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "encoder": sum(p.numel() for p in model.force_conditioner.encoder.parameters()),
            "fusion": sum(p.numel() for p in model.force_conditioner.fusion.parameters()),
        }
        collate = CustomCollate()  # cached assets only, offline env set above
        images = [Image.new("RGB", (64, 64), color) for color in ((80, 110, 130), (130, 80, 110))]
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Synthetic assembly observation. External view:"},
                {"type": "image", "image": images[0]},
                {"type": "text", "text": "TCP view:"},
                {"type": "image", "image": images[1]},
                {"type": "text", "text": "Generate assembly actions. /no_cot"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
            {"role": "user", "content": [{"type": "text", "text": "Robot state: <state>"}]},
            {"role": "assistant", "content": [{"type": "text", "text":
                "".join(f"<a_{i}>" for i in range(30)) + "<score>"}]},
        ]
        # Intentionally synthetic symmetric quantiles, never production statistics.
        raw = torch.zeros(1, 1, 60)
        raw[..., 20:26] = torch.tensor([-8., -4., -1., 2., 5., 9.])
        state = normalize_force_state(raw, torch.full((1, 60), -10.), torch.full((1, 60), 10.))
        mask = torch.zeros(30, 60, dtype=torch.int32)
        mask[:, :7] = 1
        action = torch.zeros(30, 60)
        action[:, :7] = torch.randn(30, 7) * .1  # normalized synthetic labels, no wrench
        sample = dict(messages=messages, state=state[0], action=action, action_mask=mask,
                      vlm_action_target=action, vlm_action_mask=mask, vlm_action_actual_length=30,
                      force_normalization=torch.tensor(1, dtype=torch.int64))

        def move(batch):
            return {key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()}

        train_batch = move(collate([sample]))
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4)
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        losses = model(train_batch, return_loss=True)
        if not all(torch.isfinite(value).all() for value in losses.values()):
            raise AssertionError("nonfinite synthetic loss")
        losses["loss"].backward()
        gradients = {}
        for name, parameter in model.force_conditioner.named_parameters():
            grad = parameter.grad
            if grad is None or not torch.isfinite(grad).all() or not torch.any(grad != 0):
                raise AssertionError(f"missing, nonfinite, or zero action-loss gradient: {name}")
            gradients[name] = {"norm": grad.float().norm().item(), "max": grad.float().abs().max().item()}
        report["gradients"] = gradients
        report["losses"] = {name: value.item() for name, value in losses.items()}
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del losses, train_batch

        infer = collate.processor.apply_chat_template(
            [messages[:2]], tokenize=True, return_dict=True, return_tensors="pt",
            padding=True, images_kwargs={"do_resize": False},
        )
        infer = move(dict(infer, state=state, action=torch.zeros(1, 30, 60),
                          action_mask=mask[None], force_normalization=torch.ones(1, dtype=torch.int64)))
        model.eval()

        def generate(state_override=None):
            torch.manual_seed(args.seed)
            values = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in infer.items()}
            if state_override is not None:
                values["state"] = state_override.clone()
            result = model.generate(values)
            if result.shape != (1, 30, 60) or not torch.isfinite(result).all():
                raise AssertionError("invalid synthetic generation")
            return result

        enabled = generate()
        cfg = model.force_encoder_config
        baseline_state = infer["state"].clone()
        baseline_state[..., 20:26] = 0
        try:
            model.force_encoder_config = replace(cfg, enabled=False)
            baseline = generate(baseline_state)
        finally:
            model.force_encoder_config = cfg
        report["enabled_vs_force_free_max_abs_active_action"] = (enabled[..., :7] - baseline[..., :7]).abs().max().item()
        with torch.no_grad():
            base_features = model.prepare_force_conditioning(infer["state"].bfloat16(), infer["force_normalization"])[1]
        perturbations = {}
        for axis, name in enumerate(("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")):
            changed = infer["state"].clone()
            changed[..., 20 + axis] *= -1
            with torch.no_grad():
                features = model.prepare_force_conditioning(changed.bfloat16(), infer["force_normalization"])[1]
            feature_delta = (features - base_features).abs().max().item()
            if feature_delta == 0:
                raise AssertionError(f"no encoded sensitivity to {name}")
            perturbations[name] = {"feature_max_abs": feature_delta,
                                   "active_action_max_abs": (generate(changed)[..., :7] - enabled[..., :7]).abs().max().item()}
        report["perturbations"] = perturbations
        if not any(value["active_action_max_abs"] > 0 for value in perturbations.values()):
            raise AssertionError("no observable generated-action sensitivity; inspect bfloat16/fusion scale")

        # Full model + Adam state, strict reload. This is not a ZeRO resume test.
        saved_path = args.output_dir / "synthetic_roundtrip.pt"
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "force_encoder": asdict(cfg)}, saved_path)
        expected_new = {k: v.detach().cpu().clone() for k, v in model.force_conditioner.state_dict().items()}
        with torch.no_grad():
            for parameter in model.force_conditioner.parameters():
                parameter.zero_()
        restored = torch.load(saved_path, map_location="cpu", mmap=True, weights_only=True)
        load_force_checkpoint(model, restored["state_dict"])
        restored_optimizer = torch.optim.AdamW(params, lr=1e-4)
        restored_optimizer.load_state_dict(restored["optimizer"])
        for key, value in model.force_conditioner.state_dict().items():
            torch.testing.assert_close(value.cpu(), expected_new[key], rtol=0, atol=0)
        original_states = optimizer.state_dict()["state"]
        reloaded_states = restored_optimizer.state_dict()["state"]
        if original_states.keys() != reloaded_states.keys():
            raise AssertionError("optimizer parameter states changed on reload")
        for index, values in original_states.items():
            for key, value in values.items():
                torch.testing.assert_close(value.cpu(), reloaded_states[index][key].cpu(), rtol=0, atol=0)
        torch.testing.assert_close(generate(), enabled, rtol=1e-3, atol=1e-3)
        report["strict_full_model_and_optimizer_roundtrip"] = "passed"
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        torch.cuda.synchronize(device)
        report["gpu_memory_bytes"] = {
            "peak_allocated": torch.cuda.max_memory_allocated(device),
            "peak_reserved": torch.cuda.max_memory_reserved(device),
            "allocated": torch.cuda.memory_allocated(device),
        }
        with (args.output_dir / "verification.json").open("x") as output:
            json.dump(report, output, indent=2)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
