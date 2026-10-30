"""Held-out generated-action metrics; independent of Lightning validation loops."""
from __future__ import annotations

import math
import random

import numpy as np
import torch

from mibot.utils.claix_checkpoint import rng_state, restore_rng

GROUPS = {'translation_m': slice(0, 3), 'rotation_rad': slice(3, 6), 'gripper': slice(6, 7)}


def inference_payload(item, processor):
    """Match the Doosan deployed instruction-only VLM prompt, with zero prefix.

    Dataset's final state/action turns are teacher-forcing targets and must never
    enter held-out inference. State still conditions DiT through its own projector.
    """
    messages = item['messages'][:2]
    if len(messages) != 2 or messages[1]['role'] != 'assistant':
        raise ValueError('Expected instruction and empty-CoT prompt')
    for message in messages:
        for part in message['content']:
            if part['type'] == 'text' and any(token in part['text'] for token in ('<a_', '<score>', '<state>')):
                raise ValueError('Target token leakage into validation prompt')
    payload = dict(processor.apply_chat_template([messages], tokenize=True, return_dict=True,
                                                return_tensors='pt', padding=True,
                                                images_kwargs={'do_resize': False}))
    if int(payload['image_grid_thw'].shape[0]) != 2:
        raise ValueError('Validation requires two real camera views')
    payload['state'] = item['state'].unsqueeze(0)
    payload['action'] = torch.zeros_like(item['action']).unsqueeze(0)
    payload['action_mask'] = torch.zeros_like(payload['action'], dtype=torch.int32)
    payload['action_mask'][..., :7] = 1
    return payload


def action_errors(prediction, target, mask, std):
    """Micro MAE by semantic group in physical units using training-only std.

    Mean cancels in differences. Use the source ACTION_EPS for normalization.
    Masked padded horizons and the other 53 action dimensions are excluded.
    """
    from mibot.utils.io import ACTION_EPS
    if prediction.shape != target.shape or mask.shape != target.shape or target.shape[-1] != 60:
        raise ValueError('Action/mask shape mismatch')
    valid = mask.bool()
    if not torch.isfinite(prediction[valid]).all() or not torch.isfinite(target[valid]).all():
        raise ValueError('Nonfinite held-out action')
    errors = (prediction.float() - target.float()).abs() * (std.float() + ACTION_EPS)
    result = {}
    for group, dims in GROUPS.items():
        values = errors[..., dims][valid[..., dims]]
        if not values.numel():
            raise ValueError('No valid semantic actions for held-out metric')
        result[group] = dict(absolute_error_sum=float(values.sum()), count=values.numel(),
                             mae=float(values.mean()))
    return result


def evaluate_items(model, items, processor, std, *, device='cpu', seed=73000):
    """No-grad, generated-action evaluation with per-frame deterministic seeds.

    `items` yields (episode_id, frame_index, dataset_item) in frozen order.
    Aggregation reports both frame-element micro and equal-episode macro MAE.
    """
    saved, was_training = rng_state(), model.training
    totals, episodes = {}, {}
    frames = 0
    model.eval()
    try:
        with torch.inference_mode():
            for episode, frame, item in items:
                import hashlib
                local_seed = int.from_bytes(hashlib.sha256(f'{seed}:{episode}:{frame}'.encode()).digest()[:4], 'big')
                torch.manual_seed(local_seed)
                random.seed(local_seed)
                np.random.seed(local_seed)
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in inference_payload(item, processor).items()}
                prediction = model.generate(batch).float().cpu().squeeze(0)
                metrics = action_errors(prediction, item['action'], item['action_mask'], std)
                for group, row in metrics.items():
                    for aggregate in (totals, episodes.setdefault(episode, {})):
                        acc = aggregate.setdefault(group, [0., 0])
                        acc[0] += row['absolute_error_sum']
                        acc[1] += row['count']
                frames += 1
    finally:
        model.train(was_training)
        restore_rng(saved)
    if not frames:
        raise ValueError('Empty validation evaluation')
    result = dict(frames=frames, episodes=len(episodes), micro_mae={}, episode_macro_mae={})
    for group, (value, count) in totals.items():
        result['micro_mae'][group] = value / count
        result['episode_macro_mae'][group] = sum(ep[group][0] / ep[group][1] for ep in episodes.values()) / len(episodes)
        if not math.isfinite(result['micro_mae'][group]):
            raise ValueError('Nonfinite validation metric')
    return result
