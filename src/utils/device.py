# coding: utf-8

"""
Unified device selection and MPS compatibility utilities for Apple Silicon support.
"""

import os
import torch
import contextlib
import warnings


def select_device(device_id: int = 0, flag_force_cpu: bool = False) -> str:
    """Select the best available compute device.

    Priority: CUDA > MPS (Apple Silicon) > CPU

    Args:
        device_id: CUDA device index (ignored for MPS/CPU).
        flag_force_cpu: If True, force CPU regardless of GPU availability.

    Returns:
        Device string: 'cuda:N', 'mps', or 'cpu'
    """
    if flag_force_cpu:
        return 'cpu'

    if torch.cuda.is_available():
        return f'cuda:{device_id}'

    try:
        if torch.backends.mps.is_available():
            _setup_mps_env()
            return 'mps'
    except Exception:
        pass

    return 'cpu'


def _setup_mps_env():
    """Set environment variables for MPS compatibility.

    PyTorch's MPS backend does not support every op; setting
    PYTORCH_ENABLE_MPS_FALLBACK=1 lets unsupported ops fall back
    to CPU automatically.
    """
    if 'PYTORCH_ENABLE_MPS_FALLBACK' not in os.environ:
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        warnings.warn(
            "Apple Silicon (MPS) detected. PYTORCH_ENABLE_MPS_FALLBACK=1 "
            "has been set automatically so that unsupported MPS ops fall "
            "back to CPU. You may see slight slowdowns for those ops.",
            stacklevel=3,
        )


def inference_ctx(device: str, flag_use_half_precision: bool = True) -> contextlib.AbstractContextManager:
    """Create the appropriate inference context manager.

    - CUDA: use torch.autocast with float16 if half-precision is enabled
    - MPS:  skip autocast (not fully supported), use nullcontext
    - CPU:  skip autocast

    Args:
        device: Device string from select_device()
        flag_use_half_precision: Whether to enable float16 autocast

    Returns:
        A context manager for the inference scope
    """
    if device == "mps" or device == "cpu":
        return contextlib.nullcontext()
    else:
        # device like 'cuda:0' -> device_type 'cuda'
        device_type = device.split(':')[0]
        return torch.autocast(
            device_type=device_type,
            dtype=torch.float16,
            enabled=flag_use_half_precision,
        )


def is_mps() -> bool:
    """Check if the current device is Apple Silicon MPS."""
    try:
        return torch.backends.mps.is_available()
    except Exception:
        return False


def is_cuda() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def grid_sample_3d_fallback(input: torch.Tensor, grid: torch.Tensor, **kwargs) -> torch.Tensor:
    """Perform 3D grid_sample with MPS compatibility.

    On CUDA this is a no-op wrapper around F.grid_sample.
    On MPS (Apple Silicon), grid_sample with 5D input (3D volume sampling)
    is not natively supported. We explicitly move tensors to CPU,
    run grid_sample there, and move the result back—this avoids
    the opaque fallback error and can be slightly faster than
    PYTORCH_ENABLE_MPS_FALLBACK because we batch the transfer.

    Args:
        input: 5D tensor (N, C, D_in, H_in, W_in)
        grid:  5D tensor (N, D_out, H_out, W_out, 3)
        **kwargs: Passed to F.grid_sample (e.g. align_corners, mode)

    Returns:
        5D tensor on the same device as input
    """
    import torch.nn.functional as F

    if input.device.type == 'mps':
        input_cpu = input.cpu()
        grid_cpu = grid.cpu()
        output = F.grid_sample(input_cpu, grid_cpu, **kwargs)
        return output.to('mps')
    else:
        return F.grid_sample(input, grid, **kwargs)
