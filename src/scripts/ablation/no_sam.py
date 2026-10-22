"""SPOD_DATASET_V1 NO_SAM step; unchanged C24 objective and AdamW.

Call step(training_module, model, optimizer, batch, O_step, counts, probe=False).
Parent training uses its original entry unchanged. Model construction/configure
remain SAM so parameter groups, learning rates and fixed BN are identical.
The historical B arm selects the same objective and ordinary AdamW in ct.step.
"""

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
