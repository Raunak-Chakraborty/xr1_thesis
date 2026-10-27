"""Strict original-weight validation, with an explicit all-new-module exception.

No strict=False load: validate first, then fill only the exact new parameter
names from their reproducible initialization and invoke strict=True.
"""


def force_parameter_names(module):
    return {name for name, _ in module.named_parameters()
            if name.startswith(("force_conditioner.", "model.force_conditioner."))}


def load_force_checkpoint(module, state_dict, *, allow_base_checkpoint=False):
    expected = module.state_dict()
    missing = set(expected) - set(state_dict)
    unexpected = set(state_dict) - set(expected)
    new = force_parameter_names(module)
    permitted = bool(allow_base_checkpoint and new and missing == new)
    mismatched = [name for name in set(expected) & set(state_dict)
                  if getattr(state_dict[name], "shape", None) != expected[name].shape]
    if unexpected or mismatched or (missing and not permitted):
        raise RuntimeError(
            "Checkpoint validation failed: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}, "
            f"shape_mismatch={sorted(mismatched)}. Only a complete absent force_conditioner "
            "may be initialized, with allow_base_checkpoint=true."
        )
    supplied = dict(state_dict)
    if permitted:
        supplied.update({name: expected[name] for name in missing})
    result = module.load_state_dict(supplied, strict=True)
    return result


def model_weights(checkpoint):
    """Accept plain weights, Lightning state_dict, or consolidated DeepSpeed module.

    Never filter unexpected keys away. Sharded ZeRO checkpoints must first be
    consolidated using the existing training framework's export tools.
    """
    state = checkpoint.get("module", checkpoint.get("state_dict", checkpoint))
    if state and all(key.startswith("model.") for key in state):
        return {key[len("model."):]: value for key, value in state.items()}
    return state
