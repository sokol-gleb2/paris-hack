import torch


def validate_state(state, reference):
    if not isinstance(state, dict) or set(state) != set(reference):
        raise ValueError('Candidate must be a plain state dictionary with exactly the official keys.')
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != reference[key].shape:
            raise ValueError(f'Parameter shape/type mismatch: {key}')
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f'Weights must be finite float32: {key}')
