"""Server-side counterpart of XR-1's existing numpy state normalization.

Only used for the independent encoder; generic inference is unchanged.
Statistics and raw inputs are float32, matching JsonDataset, before bfloat16.
"""

import torch

from mibot.utils.force_contract import validate_force_quantiles


def normalize_force_state(state, q01, q99):
    validate_force_quantiles(q01, q99)
    if state.ndim != 3 or state.shape[1:] != (1, 60) or not state.is_floating_point():
        raise ValueError("raw force-aware state must be floating (B,1,60)")
    if not torch.isfinite(state).all():
        raise ValueError("raw force-aware state contains nonfinite values")
    state, q01, q99 = state.float(), q01.float(), q99.float()
    if state.device != q01.device or state.device != q99.device:
        raise ValueError("state and quantile statistics must be on the same device")
    valid = q99 > q01
    result = torch.zeros_like(state)
    # Preserve numpy normalize_quantile's float32 operation ordering exactly.
    result[..., valid[0]] = (
        2.0 * ((state[..., valid[0]] - q01[..., valid[0]])
               / ((q99[..., valid[0]] + 1e-6) - q01[..., valid[0]])) - 1.0
    )
    return result.clamp(-1.0, 1.0)
