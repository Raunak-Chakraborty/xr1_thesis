"""Runtime freeze policy for Xiaomi Robotics-1, not automatically applied."""
def apply_last_n_policy(model, n):
    if n not in (1, 2, 3):
        raise ValueError('n must be 1, 2 or 3')
    stacks = {
        'vision': model.vlm.model.visual.blocks,
        'language': model.vlm.model.language_model.layers,
        'dit': model.dit.layers,
    }
    observed = {k: len(v) for k, v in stacks.items()}
    if observed != {'vision': 24, 'language': 36, 'dit': 36}:
        raise RuntimeError(f'Unrecognized XR1 architecture: {observed}')
    model.requires_grad_(False)
    for stack in stacks.values():
        for block in list(stack)[-n:]:
            block.requires_grad_(True)
    train_modules = (
        'state_projector', 'action_projector', 'action_output_layer',
        't_embedder', 't_projector', 'sink',
    )
    for name in train_modules:
        module = getattr(model, name, None)
        if module is None:
            raise RuntimeError(f'Missing XR1 robot interface module: {name}')
        module.requires_grad_(True)
    trainable = [(name, int(getattr(p, 'ds_numel', p.numel())))
                 for name, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError('Zero trainable parameters')
    return {'n': n, 'stacks': observed,
            'trainable_tensors': len(trainable),
            'trainable_parameters': sum(size for _, size in trainable),
            'trainable_names': [name for name, _ in trainable]}
