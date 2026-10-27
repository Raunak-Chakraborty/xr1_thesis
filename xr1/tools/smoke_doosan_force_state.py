"""Workstation-only one-batch checkpoint/gradient and optional inference probe.

This tool never commands a robot, steps an optimizer, or downloads weights.
Use cached Qwen assets with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="existing XR-1 model_states.pt")
    parser.add_argument("--statistics", required=True, help="real training-split statistics JSON")
    parser.add_argument("--train-path", required=True, help="converted training JSON directory")
    parser.add_argument("--output", required=True, help="new JSON report; existing files are refused")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force-free", action="store_true")
    parser.add_argument("--check-output-sensitivity", action="store_true")
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("output exists; choose a new report path")
    if not Path(args.checkpoint).is_file():
        parser.error("checkpoint must already exist locally")

    import torch
    from hydra import compose, initialize_config_dir
    from mmengine import Config, DATASETS
    from omegaconf import OmegaConf

    import mibot.data
    from mibot.models import MIMODEL
    from mibot.utils.doosan import build_doosan_action_mask
    from mibot.utils.doosan_profile import resolve_doosan_data

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("requested CUDA device is unavailable")
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
        config = compose(config_name="config", overrides=["data=doosan_force_state"])
    config.model.params.pretrained = str(Path(args.checkpoint).resolve())
    data = config.data.params.train_datasets
    data.paths = [str(Path(args.train_path).resolve())]
    data.normalization_path = str(Path(args.statistics).resolve())
    data.use_wrench = not args.force_free
    data.batch_size = 1
    data.num_workers = 0
    data.persistent_workers = False
    config.data.params.max_steps = 1
    cfg = Config(OmegaConf.to_container(config, resolve=True))
    cfg.data.params.train_datasets = resolve_doosan_data(cfg.data.params.train_datasets)

    random.seed(42)
    torch.manual_seed(42)
    datamodule = DATASETS.build(cfg.data)
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    batch = {key: value.to(args.device) if isinstance(value, torch.Tensor) else value
             for key, value in batch.items()}
    state = batch["state"].detach().float().requires_grad_(True)
    batch["state"] = state
    runner = MIMODEL.build(cfg.model)
    runner.configure_model()
    model = runner.model.to(args.device).train()
    losses = model(copy.copy(batch), return_loss=True)
    if not all(torch.isfinite(value).all() for value in losses.values()):
        raise RuntimeError("non-finite one-batch loss")
    losses["loss"].backward()
    if state.grad is None or not torch.isfinite(state.grad).all():
        raise RuntimeError("state input gradients are absent or non-finite")
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "use_wrench": not args.force_free,
        "losses": {key: float(value.detach().float()) for key, value in losses.items()},
        "normalized_wrench": state.detach()[0, 0, 20:26].cpu().tolist(),
        "wrench_input_grad_abs": state.grad[..., 20:26].abs().sum(dim=(0, 1)).cpu().tolist(),
        "normalization_metadata": cfg.data.params.train_datasets.get("normalization_metadata"),
        "optimizer_steps": 0,
    }
    for name in ("state_projector_choice", "state_projector"):
        gradient = getattr(model, name).layers[0].weight.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError(f"{name} gradients are absent or non-finite")
        report[name + "_wrench_column_grad_norm"] = gradient[:, 20:26].float().norm(dim=0).cpu().tolist()

    model.zero_grad(set_to_none=True)
    if args.check_output_sensitivity:
        model.eval()
        sample = datamodule.train_set[0]
        inference = datamodule.collate_fn.processor.apply_chat_template(
            [sample["messages"][:-2]], tokenize=True, return_dict=True, return_tensors="pt",
            padding=True, images_kwargs={"do_resize": False},
        )
        inference = {key: value.to(args.device) for key, value in inference.items()}
        inference.update(
            state=sample["state"][None].to(args.device),
            action=torch.zeros((1, 30, 60), device=args.device, dtype=torch.bfloat16),
            action_mask=torch.from_numpy(build_doosan_action_mask(30))[None].to(args.device),
        )

        def generate(payload):
            torch.manual_seed(42)
            return model.generate(copy.copy(payload)).float()

        with torch.no_grad():
            reference = generate(inference)
            if not torch.isfinite(reference).all():
                raise RuntimeError("non-finite reference inference")
            differences = []
            for index in range(20, 26):
                changed = copy.copy(inference)
                changed["state"] = inference["state"].clone()
                value = changed["state"][0, 0, index]
                changed["state"][0, 0, index] = -0.25 if abs(float(value) - 0.25) < 0.1 else 0.25
                predicted = generate(changed)
                if not torch.isfinite(predicted).all():
                    raise RuntimeError("non-finite inference probe")
                differences.append(float((predicted[..., :7] - reference[..., :7]).abs().max()))
        report["normalized_wrench_perturbation_action_max_abs_diff"] = differences
        report["sensitivity_note"] = "Direct normalized-state probe; force-free deployment masks these slots."
    with Path(args.output).open("x", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2)
        destination.write("\n")
    print(f"wrote {args.output}; one backward pass, zero optimizer steps, no robot commands")


if __name__ == "__main__":
    main()
