"""Dependency-free configuration and normalization contract for experiment B.

The model consumes already quantile-normalized Doosan BASE wrench observations,
ordered Fx, Fy, Fz, Tx, Ty, Tz. No physical compensation or contact labels.
"""

from dataclasses import dataclass
import math

WRENCH_START = 20
WRENCH_STOP = 26
WRENCH_CHANNELS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
FORCE_NORMALIZATION_ID = 1  # XR-1 state quantiles, clipped to [-1, 1], once


@dataclass(frozen=True)
class ForceEncoderConfig:
    enabled: bool = False
    hidden_dims: tuple = (64, 64)
    output_dim: int = 64
    fusion_type: str = "adaln_residual"
    fusion_init: str = "small_normal"
    fusion_init_std: float = 0.001
    init_seed: int = 42
    dual_path: bool = False
    pretrained_policy: str = "original"
    train_encoder: bool = True
    train_fusion: bool = True
    input_normalization: str = "state_quantile_once"

    @classmethod
    def from_mapping(cls, value=None):
        if isinstance(value, cls):
            return value
        config = cls(**dict(value or {}))
        for key in ("enabled", "dual_path", "train_encoder", "train_fusion"):
            if not isinstance(getattr(config, key), bool):
                raise ValueError(f"force_encoder.{key} must be a boolean")
        if not isinstance(config.hidden_dims, (list, tuple)):
            raise ValueError("force_encoder.hidden_dims must be a list of positive widths")
        for width in (*config.hidden_dims, config.output_dim):
            if type(width) is not int or width <= 0:
                raise ValueError("force encoder dimensions must be positive integers")
        if config.fusion_type != "adaln_residual":
            raise ValueError("only fusion_type=adaln_residual is implemented")
        if config.fusion_init not in {"small_normal", "zero"}:
            raise ValueError("fusion_init must be small_normal or zero")
        if not math.isfinite(config.fusion_init_std) or config.fusion_init_std <= 0:
            raise ValueError("fusion_init_std must be finite and positive")
        if type(config.init_seed) is not int or not 0 <= config.init_seed < 2**63:
            raise ValueError("init_seed must be an integer in [0, 2**63)")
        if config.pretrained_policy not in {"original", "frozen", "dit", "last_vlm_block"}:
            raise ValueError("unknown pretrained_policy")
        if config.input_normalization != "state_quantile_once":
            raise ValueError("force encoder requires input_normalization=state_quantile_once")
        if config.enabled and config.fusion_init == "zero" and config.train_encoder and not config.train_fusion:
            raise ValueError("a frozen zero fusion would permanently disconnect the encoder")
        return config


def validate_force_quantiles(q01, q99):
    """Reject missing/padded/nonfinite/degenerate statistics for any wrench axis."""
    arrays = []
    for name, value in (("q01", q01), ("q99", q99)):
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or len(value) != 1:
            raise ValueError(f"force {name} requires XR-1-layout statistics of shape (1,60)")
        if not isinstance(value[0], (list, tuple)) or len(value[0]) != 60:
            raise ValueError(f"force {name} requires XR-1-layout statistics of shape (1,60)")
        try:
            row = [float(x) for x in value[0]]
        except (TypeError, ValueError) as error:
            raise ValueError(f"force {name} must contain numeric statistics") from error
        if not all(math.isfinite(x) for x in row):
            raise ValueError(f"force {name} contains nonfinite statistics")
        arrays.append(row)
    for index, name in enumerate(WRENCH_CHANNELS, WRENCH_START):
        if arrays[1][index] <= arrays[0][index]:
            raise ValueError(f"wrench {name} at state[{index}] requires finite q99 > q01; zero padding is not a statistic")
    return q01, q99


def validate_force_experiment(model_config, data_config):
    """Fail before model construction if experiment B is wired to the wrong data."""
    config = ForceEncoderConfig.from_mapping(model_config.get("force_encoder"))
    if not config.enabled:
        return
    if data_config.get("type") != "DoosanDataModule":
        raise ValueError("independent force encoder requires DoosanDataModule")
    data = data_config["params"]["train_datasets"]
    if data.get("use_wrench") is not True or data.get("force_encoder_input") is not True:
        raise ValueError("independent encoder requires use_wrench=true and force_encoder_input=true")
    if data.get("action_length", 30) != 30:
        raise ValueError("independent encoder experiment requires action_length=30")
    validate_force_quantiles(data.get("q01"), data.get("q99"))
