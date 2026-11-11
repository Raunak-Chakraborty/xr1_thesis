import sys
import os
import gc
import json
import copy
import random
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor

policy, selection_file, valroot, assets, work, outroot = sys.argv[1:]
valroot = Path(valroot)
assets = Path(assets)
work = Path(work)
outroot = Path(outroot)
incoming = Path(os.environ["XR1_NEW_JOB1_ROOT"])

torch.set_num_threads(4)

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

with open(selection_file) as f:
    selection = json.load(f)

windows = selection["windows"]
require(len(windows) == 476, "Expected 476 frozen evaluation predictions")
require(selection["split"] == "val", "Wrong evaluation split")
require(selection["action_horizon"] == 30, "Wrong horizon")

details = {
    "last_one": {"step": 48129, "last_n": 1,
       "checkpoint": incoming / "parent/step-048129.ckpt",
       "config": work / "metadata/metadata/last_one/resolved_compact_config.json"},
    "last_two": {"step": 47501, "last_n": 2,
       "checkpoint": assets / "checkpoints/last_two/step-047501.ckpt",
       "config": None},
    "adamw_control": {"step": 500, "last_n": 1,
       "checkpoint": incoming / "profiles/adamw_control/motion-000500.ckpt",
       "config": incoming / "config/resolved_compact_config.json"},
    "muon_conservative": {"step": 500, "last_n": 1,
       "checkpoint": incoming / "profiles/muon_conservative/motion-000500.ckpt",
       "config": incoming / "config/resolved_compact_config.json"},
    "muon_faster": {"step": 500, "last_n": 1,
       "checkpoint": incoming / "profiles/muon_faster/motion-000500.ckpt",
       "config": incoming / "config/resolved_compact_config.json"},
}
NEW_POLICIES = {"adamw_control", "muon_conservative", "muon_faster"}

require(policy in details, "Unknown policy")
info = details[policy]

if policy == "last_two":
    found = sorted(
        (work / "metadata/metadata/last_two").glob(
            "resolved-compact-*.json"
        )
    )
    require(len(found) == 1, "Ambiguous last-two resolved config")
    info["config"] = found[0]

cfg = json.loads(info["config"].read_text())

print(f"\n===== MODEL {policy.upper()} =====", flush=True)
print("SOURCE_POLICY=", policy)
print("CHECKPOINT_STEP=", info["step"])

train_data = copy.deepcopy(
    cfg["data"]["params"]["train_datasets"]
)

require(train_data["action_length"] == 30, "Horizon drift")
require(train_data["use_wrench"] is False, "Force policy drift")
require(
    train_data["semantic_action_key"] == "actions.semantic",
    "Action semantics drift",
)

# Both models must use the same frozen training statistics.
reference = json.loads(
    (
        work / "metadata/metadata/last_one/resolved_compact_config.json"
    ).read_text()
)
reference_data = reference["data"]["params"]["train_datasets"]

for name in ("mean", "std", "q01", "q99"):
    require(
        np.array_equal(
            np.asarray(train_data[name]),
            np.asarray(reference_data[name]),
        ),
        f"Frozen normalization differs: {name}",
    )

print("FROZEN_NORMALIZATION=PASS", flush=True)

train_data["paths"] = [str(valroot)]
train_data["batch_size"] = 1
train_data["num_workers"] = 0
train_data["deterministic_replay"] = False

dataset_cfg = {
    "max_steps": 1,
    "train_datasets": train_data,
}

from mibot.data.datasets.doosan_json_dataset import DoosanValidationDataset
from mibot.data.collate import custom_collate
from mibot.utils.io import denormalize_action
from mibot.models.VLA.XR1 import xr1
from mibot.utils.doosan_last_n import apply_last_n_policy

processor_path = (
    work / "processor/processor/Qwen3-VL-4B-Instruct"
)

OriginalProcessor = AutoProcessor

class LocalProcessor:
    @staticmethod
    def from_pretrained(identifier, *args, **kwargs):
        require(
            identifier == "Qwen/Qwen3-VL-4B-Instruct",
            "Wrong processor identifier",
        )
        kwargs["local_files_only"] = True
        return OriginalProcessor.from_pretrained(
            str(processor_path),
            *args,
            **kwargs,
        )

custom_collate.AutoProcessor = LocalProcessor

dataset = DoosanValidationDataset(dataset_cfg)
require(
    len(dataset) == selection["validation_frame_count"],
    "Validation dataset index mapping changed",
)

collator = custom_collate.CustomCollate(
    strict_samples=True,
    expected_views=2,
)

prepared = []

print("PREPARING_SAMPLES=START", flush=True)

for index, window in enumerate(windows):
    sample_index = int(window["sample_index"])
    frame_index = int(window["frame_index"])

    record = dataset.samples[sample_index]

    actual_episode = (
        Path(record["file"]).relative_to(valroot).as_posix()
    )

    require(
        actual_episode == window["episode_relative"]
        and int(record["frame_index"]) == frame_index,
        f"Frozen sample mismatch at window {index+1}",
    )

    random.seed(int(window["prompt_seed"]))
    sample = dataset[sample_index]
    messages = sample["messages"][:2]
    require(len(messages) == 2 and messages[1]["role"] == "assistant",
            "Expected clean inference prompt")
    for message in messages:
        for part in message["content"]:
            if part["type"] == "text":
                require(not any(token in part["text"] for token in ("<a_", "<score>", "<state>")),
                        "Teacher-forced action-token leakage in inference")
    batch = dict(collator.processor.apply_chat_template(
        [messages], tokenize=True, return_dict=True,
        return_tensors="pt", padding=True,
        images_kwargs={"do_resize": False},
    ))
    batch["state"] = sample["state"].unsqueeze(0)
    batch["action"] = torch.zeros_like(sample["action"]).unsqueeze(0)
    batch["action_mask"] = torch.zeros_like(batch["action"], dtype=torch.int32)
    batch["action_mask"][..., :7] = 1

    require(batch["action"].shape == (1, 30, 60), "Action shape")
    require(batch["state"].shape == (1, 1, 60), "State shape")

    grid = batch.get("image_grid_thw")
    require(grid is not None and grid.shape[0] == 2,
            "Camera views missing")

    input_hash = hashlib.sha256(
        batch["input_ids"].numpy().tobytes()
    ).hexdigest()

    pixel_hash = hashlib.sha256(
        batch["pixel_values"].numpy().tobytes()
    ).hexdigest()

    prepared.append((index, window, sample, batch,
                     input_hash, pixel_hash))

    print(
        f"SAMPLE_{index+1}=READY "
        f"frame={frame_index} "
        f"tokens={batch['input_ids'].shape[1]}",
        flush=True,
    )

print("PREPARING_SAMPLES=PASS", flush=True)

require(torch.cuda.is_available(), "CUDA missing")
free, total = torch.cuda.mem_get_info(0)

print(f"GPU_FREE_BEFORE_GIB={free/2**30:.2f}", flush=True)
require(free >= 16 * 2**30, "Insufficient free GPU memory")

print("MODEL_CONSTRUCTION=START", flush=True)

model_cfg = copy.deepcopy(
    cfg["model"]["params"]["model"]
)
require(model_cfg.pop("type") == "xr1", "Wrong architecture")
if policy in NEW_POLICIES:
    model_cfg["motion_focus_loss"] = {"first_5": 2.0, "next_5": 1.25, "remaining_20": 0.75}

model = xr1(**model_cfg)

print("MODEL_CONSTRUCTION=PASS", flush=True)

base = torch.load(
    assets / "shared/pretrained/model_states.pt",
    map_location="cpu",
    mmap=True,
    weights_only=False,
)

require(set(base) == {"module"}, "Unexpected baseline format")

base_state = {
    k.removeprefix("model."): v
    for k, v in base["module"].items()
}

model.load_state_dict(base_state, strict=True)

del base, base_state
gc.collect()

print("ORIGINAL_BASE_RESTORED=PASS", flush=True)

selection_report = apply_last_n_policy(
    model,
    info["last_n"],
)

selected = set(selection_report["trainable_names"])

checkpoint = torch.load(
    info["checkpoint"],
    map_location="cpu",
    mmap=True,
    weights_only=False,
)

state = checkpoint["state_dict"]
require(checkpoint["global_step"] == info["step"], "Checkpoint step mismatch")
expected_base_sha = "94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7"
expected_lang_sha = "21d8ffbdbfec2f9833339faaf0c5818c85cd5a135dcf474fe2d3f76cda1cf34d"

if policy in NEW_POLICIES:
    meta = checkpoint["xr1_motion_compact"]
    require(meta["schema"] == "xr1-motion-compact-lightning-v1", "Wrong experimental schema")
    require(meta["optimizer_kind"] == ("adamw_control" if policy == "adamw_control" else "hybrid_muon"), "Optimizer provenance mismatch")
    require(meta["parent"]["parent_step"] == 48129, "Parent step mismatch")
    require(meta["parent"]["pretrained_sha256"] == expected_base_sha, "Pretrained provenance mismatch")
    require(meta["parent"]["language_sha256"] == expected_lang_sha, "Language provenance mismatch")
    require(meta["parent"]["selected_tensor_count"] == 45, "Parent tensor count mismatch")
    require(meta["parent"]["trainable_elements"] == 141255168, "Parent element count mismatch")
    require(checkpoint["xr1_motion_replay"]["global_step"] == 500, "Replay cursor mismatch")
    require(bool(checkpoint.get("optimizer_states")) and bool(checkpoint.get("lr_schedulers")), "Missing resumable checkpoint optimizer/scheduler")
else:
    meta = checkpoint["xr1_compact_fp32"]
    require(meta["last_n"] == info["last_n"], "Checkpoint last-N mismatch")
    require(meta["pretrained_sha256"] == expected_base_sha, "Baseline provenance mismatch")
    require(meta["language_sha256"] == expected_lang_sha, "Language provenance mismatch")

expected_full = {"model." + k for k in selected}

require(set(state) == expected_full,
        "Fine-tuned parameter selection mismatch")

key_hash = hashlib.sha256(
    "\n".join(sorted(expected_full)).encode()
).hexdigest()
require(
    key_hash == (meta["selected_keys_sha256"] if policy in NEW_POLICIES else meta["selected_key_sha256"]),
    "Fine-tuned key fingerprint mismatch",
)

parameters = dict(model.named_parameters())

require(
    sum(parameters[k].numel() for k in selected)
    == (meta["parent"]["trainable_elements"] if policy in NEW_POLICIES else meta["trainable_parameter_count"]),
    "Fine-tuned element count mismatch",
)

if policy in NEW_POLICIES:
    require(meta["parent"]["selected_key_sha256"] == key_hash, "Parent trainable-key mismatch")
    require(len(selected) == 45, "New candidate tensor count mismatch")
for name in selected:
    parameters[name].data = (
        parameters[name].detach().float().contiguous()
    )

partial = {
    name.removeprefix("model."): tensor
    for name, tensor in state.items()
}

loaded = model.load_state_dict(partial, strict=False)
require(not loaded.unexpected_keys, "Unexpected checkpoint keys")
require(
    set(loaded.missing_keys) == set(model.state_dict()) - selected,
    "Partial checkpoint load coverage mismatch",
)

for name in selected:
    require(
        parameters[name].dtype == torch.float32,
        f"FP32 master weight lost: {name}",
    )

del checkpoint, partial, state, loaded
gc.collect()

model.requires_grad_(False)
model.eval()
model.to("cuda:0")

torch.cuda.synchronize()

print(
    f"FINETUNED_RESTORE=PASS "
    f"tensors={len(selected)} "
    f"GPU_ALLOCATED_GIB={torch.cuda.memory_allocated()/2**30:.3f}",
    flush=True,
)

outdir = outroot / policy
outdir.mkdir(parents=True, exist_ok=False)

rows = []

for index, window, sample, batch, token_hash, pixel_hash in prepared:
    seed = int(window["seed"])

    inputs = {
        k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.cuda.synchronize()
    start = time.monotonic()

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = model.generate(dict(inputs))

    torch.cuda.synchronize()
    elapsed = time.monotonic() - start

    require(
        predicted.shape == (1, 30, 60)
        and torch.isfinite(predicted).all().item(),
        "Invalid model prediction",
    )

    pnorm = predicted.float().cpu().numpy()[0]
    gnorm = sample["action"].numpy()

    pred = denormalize_action(
        pnorm, dataset.mean, dataset.std
    ).astype(np.float64)

    truth = denormalize_action(
        gnorm, dataset.mean, dataset.std
    ).astype(np.float64)

    baseline = np.asarray(dataset.mean, dtype=np.float64)

    mask = sample["action_mask"].numpy().astype(bool)
    valid = mask[:, 0]
    require(valid.any(), "No valid action steps")

    delta = pred[valid, :7] - truth[valid, :7]
    baseline_delta = (
        baseline[valid, :7] - truth[valid, :7]
    )

    translation_mae = float(
        np.linalg.norm(delta[:, :3], axis=1).mean() * 1000
    )
    rotation_mae = float(
        np.linalg.norm(delta[:, 3:6], axis=1).mean()
        * 180 / np.pi
    )
    gripper_mae = float(np.abs(delta[:, 6]).mean())

    baseline_translation = float(
        np.linalg.norm(
            baseline_delta[:, :3], axis=1
        ).mean() * 1000
    )
    baseline_rotation = float(
        np.linalg.norm(
            baseline_delta[:, 3:6], axis=1
        ).mean() * 180 / np.pi
    )
    baseline_gripper = float(
        np.abs(baseline_delta[:, 6]).mean()
    )

    row = {
        "category": window["selection_category"],
        "suite": window["suite"],
        "repeat": int(window["repeat"]),
        "episode": window["episode_relative"],
        "frame_index": window["frame_index"],
        "sample_index": window["sample_index"],
        "seed": seed,
        "input_token_sha256": token_hash,
        "input_pixel_sha256": pixel_hash,
        "latency_seconds": elapsed,
        "first5_normalized_motion_mae": float(np.abs(pnorm[:5,:6] - gnorm[:5,:6]).mean()),
        "first5_translation_error_mm": float(np.linalg.norm(delta[:min(5,len(delta)),:3], axis=1).mean()*1000),
        "first5_rotation_error_deg": float(np.linalg.norm(delta[:min(5,len(delta)),3:6], axis=1).mean()*180/np.pi),
        "net_translation_error_mm": float(np.linalg.norm(delta[:,:3].sum(axis=0))*1000),
        "predicted_translation_mean_mm": float(np.linalg.norm(pred[valid,:3],axis=1).mean()*1000),
        "target_translation_mean_mm": float(np.linalg.norm(truth[valid,:3],axis=1).mean()*1000),
        "predicted_rotation_mean_deg": float(np.linalg.norm(pred[valid,3:6],axis=1).mean()*180/np.pi),
        "target_rotation_mean_deg": float(np.linalg.norm(truth[valid,3:6],axis=1).mean()*180/np.pi),
        "target_gripper_open_fraction": float(np.mean(truth[valid,6] >= .5)),
        "predicted_gripper_open_fraction": float(np.mean(pred[valid,6] >= .5)),
        "translation_error_mm": translation_mae,
        "rotation_error_deg": rotation_mae,
        "gripper_error": gripper_mae,
        "baseline_translation_error_mm": baseline_translation,
        "baseline_rotation_error_deg": baseline_rotation,
        "baseline_gripper_error": baseline_gripper,
        "target_gripper_range": float(
            np.ptp(truth[valid, 6])
        ),
        "predicted_gripper_range": float(
            np.ptp(pred[valid, 6])
        ),
    }

    rows.append(row)

    np.savez_compressed(
        outdir / f"window_{index+1:02d}.npz",
        prediction_normalized=pnorm,
        target_normalized=gnorm,
        prediction_physical=pred,
        target_physical=truth,
        action_mask=mask,
        sample_index=window["sample_index"],
        seed=seed,
    )

    print(
        f"WINDOW_{index+1}=PASS "
        f"category={row['category']} "
        f"translation_mm={translation_mae:.5f} "
        f"rotation_deg={rotation_mae:.5f} "
        f"gripper={gripper_mae:.5f} "
        f"latency_s={elapsed:.3f}",
        flush=True,
    )

with (outdir / "metrics.json").open("x") as f:
    json.dump(
        {
            "model": policy,
            "checkpoint_step": info["step"],
            "windows": rows,
        },
        f,
        indent=2,
    )
    f.write("\n")

print(f"MODEL_{policy.upper()}_EVALUATION=PASS", flush=True)
print("INFERENCE_PROMPT_TARGET_LEAKAGE=NO", flush=True)
print(f"RESULTS={outdir}", flush=True)
print(
    f"GPU_PEAK_GIB={torch.cuda.max_memory_allocated()/2**30:.3f}",
    flush=True,
)
