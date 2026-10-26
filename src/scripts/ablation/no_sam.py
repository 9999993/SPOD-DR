"""NO_SAM training step with the shared objective and AdamW.

Retain the SAM model configuration for parameter groups, learning rates and
fixed BN statistics; use arm B during the update to disable the SAM step."""

def step(training_module, model, optimizer, batch, local, counts, probe=False):
    original_arm = model.arm
    assert original_arm == 'SAM', original_arm
    previous_updates = counts['updates']
    previous_forwards = counts['training_forward']
    try:
        model.arm = 'B'
        result = training_module.step(model, optimizer, batch, local, counts, probe)
    finally:
        model.arm = original_arm
    assert not result['SAM']['active']
    assert result['SAM']['second_D'] is None
    assert result['SAM']['perturbation_norm'] == 0
    assert counts['updates'] - previous_updates == 1
    assert counts['training_forward'] - previous_forwards == 2 * len(batch['grade'])
    return result
