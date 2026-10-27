#!/usr/bin/env python3
"""Import preflight for the persistent encoder CPU test environment.

Imports model definitions but never constructs XR1, loads weights, or downloads
assets. Missing dependencies fail explicitly instead of silently skipping tests.
"""

import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    modules = (
        "torch", "torchvision", "numpy", "pytest", "transformers", "lightning",
        "hydra", "omegaconf", "mmengine", "PIL", "tqdm", "decord", "triton",
        "liger_kernel.transformers.rms_norm", "liger_kernel.transformers.rope",
        "mibot.data", "mibot.models.VLA.XR1", "mibot.server.runtime.doosan_client",
    )
    report = {"python_executable": sys.executable, "python_version": sys.version,
              "venv": sys.prefix, "imports": {}, "errors": {}}
    for name in modules:
        try:
            importlib.import_module(name)
            report["imports"][name] = "passed"
        except Exception as error:
            report["errors"][name] = f"{type(error).__name__}: {error}"
    packages = sorted(metadata.distributions(), key=lambda item: item.metadata["Name"].lower())
    report["packages"] = {item.metadata["Name"]: item.version for item in packages}
    nvidia = [name for name in report["packages"] if name.lower().startswith("nvidia-")]
    if nvidia:
        report["errors"]["nvidia_packages"] = nvidia
    if "torch" not in report["errors"]:
        import torch
        report["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                           "cuda_available": torch.cuda.is_available(),
                           "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}
        if torch.version.cuda is not None or torch.cuda.is_available():
            report["errors"]["cpu_only"] = "A CPU-only PyTorch distribution is required"
        # Execute small native CPU operations as well as checking imports.
        try:
            for dtype in (torch.float32, torch.bfloat16):
                value = torch.ones(2, 6, dtype=dtype, requires_grad=True)
                value.square().mean().backward()
                if value.grad is None or not torch.isfinite(value.grad).all() or not torch.any(value.grad != 0):
                    raise AssertionError(f"invalid {dtype} CPU gradient")
            report["cpu_fp32_bf16_autograd"] = "passed"
        except Exception as error:
            report["errors"]["cpu_autograd"] = f"{type(error).__name__}: {error}"
    report["status"] = "passed" if not report["errors"] else "failed"
    print(json.dumps(report, indent=2))
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
