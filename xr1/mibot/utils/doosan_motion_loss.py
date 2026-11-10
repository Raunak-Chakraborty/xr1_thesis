"""Optional 30-step Cartesian-motion emphasis for Doosan XR-1 flow loss.

Only motion dimensions 0..5 are reweighted. Neither target actions nor
validation inference behavior is changed. Disabled by default.
"""
import math

import torch

_KEYS = ('first_5', 'next_5', 'remaining_20')


def validate_motion_profile(profile):
    if profile is None:
        return None
    if not hasattr(profile, 'items') or set(profile) != set(_KEYS):
        raise ValueError(
            'motion_focus_loss requires first_5, next_5, remaining_20'
        )
    values = tuple(float(profile[k]) for k in _KEYS)
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError(
            'motion_focus_loss weights must be positive finite numbers'
        )
    return values


def motion_multipliers(mask, *, prefix_length, profile):
    """Return per-element weights using original horizon positions."""
    if mask.ndim != 3 or mask.shape[2] != 60:
        raise ValueError('Expected [batch, horizon, 60] action mask')
    if not 0 <= int(prefix_length) <= 30 or mask.shape[1] + prefix_length != 30:
        raise ValueError('Invalid 30-step horizon/prefix length')
    if profile is None:
        raise ValueError('Motion profile is disabled')

    a, b, c = profile

    pos = (
        torch.arange(mask.shape[1], device=mask.device)
        + int(prefix_length)
    )

    scalar = torch.where(
        pos < 5, a,
        torch.where(pos < 10, b, c)
    )

    out = torch.ones(
        mask.shape,
        device=mask.device,
        dtype=torch.float32
    )

    out[:, :, :6] = scalar[None, :, None]
    return out


def masked_weighted_mean(error, valid_mask, multiplier=None):
    """Default matches original masked mean."""
    if error.shape != valid_mask.shape:
        raise ValueError('Mismatched error/action-mask shape')

    valid = valid_mask.bool()

    if not bool(valid.any()):
        return error.sum() * 0.0

    if multiplier is None:
        return error[valid].mean()

    if multiplier.shape != error.shape:
        raise ValueError('Mismatched motion-focus multiplier shape')

    weights = multiplier[valid].to(dtype=error.dtype)

    return (
        (error[valid] * weights).sum()
        / weights.sum()
    )
