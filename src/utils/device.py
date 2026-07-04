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
    is not natively supported. We move data to CPU, compute there,
    and move the result back to MPS.

    The CPU computation is robust: we create fresh CPU tensors via
    .cpu().clone().float() to avoid MPS memory sync issues
    that can cause segfaults on some macOS configurations.
    An extra try/except guard catches residual MPS transfer failures
    and falls back to creating entirely new CPU tensors from numpy copies.

    Args:
        input: 5D tensor (N, C, D_in, H_in, W_in)
        grid:  5D tensor (N, D_out, H_out, W_out, 3)
        **kwargs: Passed to F.grid_sample (e.g. align_corners, mode)

    Returns:
        5D tensor on the same device as input
    """
    import torch.nn.functional as F

    if input.device.type != 'mps':
        return F.grid_sample(input, grid, **kwargs)

    # --- MPS path: compute grid_sample on CPU, return result to MPS ---
    orig_device = input.device

    # Strategy 1: .cpu().clone().float() — copy data to CPU first, then
    # clone & cast to ensure no shared MPS storage.  (Putting .clone()
    # after .cpu() avoids touching MPS memory after the copy.)
    try:
        input_cpu = input.cpu().clone().float()
        grid_cpu = grid.cpu().clone().float()
        output = F.grid_sample(input_cpu, grid_cpu, **kwargs)
        return output.to(orig_device)
    except Exception as exc1:
        warnings.warn(
            f"grid_sample_3d_fallback: .cpu() transfer failed ({exc1}); "
            "trying numpy round-trip fallback.",
            stacklevel=2,
        )

    # Strategy 2: numpy round-trip — completely sidestep PyTorch's
    # MPS → CPU transfer path by going through NumPy, which uses
    # a different memory copy mechanism.
    try:
        input_cpu = torch.from_numpy(input.cpu().numpy()).float()
        grid_cpu = torch.from_numpy(grid.cpu().numpy()).float()
        output = F.grid_sample(input_cpu, grid_cpu, **kwargs)
        return output.to(orig_device)
    except Exception as exc2:
        warnings.warn(
            f"grid_sample_3d_fallback: numpy round-trip also failed ({exc2}); "
            "falling back to pure CPU result (not returning to MPS).",
            stacklevel=2,
        )

    # Strategy 3: Give up returning to MPS; compute on CPU and stay there.
    # This means callers downstream will see a CPU tensor, but at least
    # we avoid a crash.  Most pipelines handle mixed devices gracefully.
    try:
        input_cpu = input.cpu().clone().float()
        grid_cpu = grid.cpu().clone().float()
        return F.grid_sample(input_cpu, grid_cpu, **kwargs)
    except Exception:
        # Last resort: try with contiguous clones
        input_cpu = input.cpu().contiguous().clone().float()
        grid_cpu = grid.cpu().contiguous().clone().float()
        return F.grid_sample(input_cpu, grid_cpu, **kwargs)
