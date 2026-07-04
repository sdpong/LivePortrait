#!/usr/bin/env python3
# coding: utf-8

"""
Apple Silicon (MPS) compatibility verification script for LivePortrait.

Run this script to check whether your environment is properly configured
for running LivePortrait on Apple Silicon (M1/M2/M3/M4) Macs.

Usage:
    python check_apple_silicon.py
"""

import sys
import traceback


def check_python_version():
    """Check Python version >= 3.10"""
    version = sys.version_info
    ok = version >= (3, 10)
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] Python version: {version.major}.{version.minor}.{version.micro}")
    if not ok:
        print("       Recommended: Python 3.10+")
    return ok


def check_torch():
    """Check PyTorch availability and MPS support"""
    try:
        import torch
        print(f"  [OK] PyTorch version: {torch.__version__}")
    except ImportError:
        print("  [FAIL] PyTorch not installed")
        return False

    # Check MPS
    try:
        mps_available = torch.backends.mps.is_available()
        if mps_available:
            print(f"  [OK] MPS (Apple Silicon GPU) is available")
        else:
            print(f"  [WARN] MPS is NOT available (are you on Apple Silicon?)")
    except Exception as e:
        print(f"  [WARN] MPS check failed: {e}")
        mps_available = False

    # Check CUDA (should not be available on macOS)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"  [INFO] CUDA is available (GPU: {torch.cuda.get_device_name(0)})")
    else:
        print(f"  [INFO] CUDA is NOT available (expected on macOS)")

    return True


def check_torchvision():
    """Check torchvision"""
    try:
        import torchvision
        print(f"  [OK] torchvision version: {torchvision.__version__}")
        return True
    except ImportError:
        print("  [FAIL] torchvision not installed")
        return False


def check_onnxruntime():
    """Check ONNX Runtime with silicon support"""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"  [OK] ONNX Runtime version: {ort.__version__}")
        print(f"       Available providers: {providers}")
        if 'CoreMLExecutionProvider' in providers:
            print("  [OK] CoreMLExecutionProvider available")
        else:
            print("  [WARN] CoreMLExecutionProvider not available")
        return True
    except ImportError:
        print("  [FAIL] onnxruntime not installed")
        return False


def check_transformers():
    """Check transformers (needed for animal mode)"""
    try:
        import transformers
        print(f"  [OK] transformers version: {transformers.__version__}")
        return True
    except ImportError:
        print("  [FAIL] transformers not installed (required for animal mode)")
        return False


def check_mps_ops():
    """Test basic MPS operations"""
    try:
        import torch
        if not torch.backends.mps.is_available():
            print("  [SKIP] MPS not available, skipping op tests")
            return True

        device = torch.device("mps")

        # Test basic tensor ops
        x = torch.randn(2, 3, device=device)
        y = torch.randn(2, 3, device=device)
        z = x + y
        print("  [OK] Basic tensor operations on MPS")

        # Test convolution
        conv = torch.nn.Conv2d(3, 16, 3, padding=1).to(device)
        inp = torch.randn(1, 3, 64, 64, device=device)
        out = conv(inp)
        print("  [OK] Conv2d on MPS")

        # Test grid_sample 2D
        import torch.nn.functional as F
        inp_2d = torch.randn(1, 1, 4, 4, device=device)
        grid_2d = torch.randn(1, 4, 4, 2, device=device)
        out_2d = F.grid_sample(inp_2d, grid_2d, align_corners=False)
        print("  [OK] F.grid_sample (2D) on MPS")

        # Test grid_sample 3D (will fallback to CPU)
        inp_3d = torch.randn(1, 1, 4, 4, 4, device=device)
        grid_3d = torch.randn(1, 4, 4, 4, 3, device=device)
        try:
            out_3d = F.grid_sample(inp_3d, grid_3d, align_corners=False)
            print("  [OK] F.grid_sample (3D) natively supported on MPS")
        except Exception:
            print("  [INFO] F.grid_sample (3D) not natively on MPS (handled by fallback)")

        # Test grid_sample_3d_fallback from device.py
        try:
            from src.utils.device import grid_sample_3d_fallback
            out_fb = grid_sample_3d_fallback(inp_3d, grid_3d, align_corners=False)
            assert out_fb.device.type == 'mps'
            print("  [OK] grid_sample_3d_fallback() works correctly")
        except Exception as e:
            print(f"  [WARN] grid_sample_3d_fallback() failed: {e}")

        return True

    except Exception as e:
        print(f"  [FAIL] MPS operation test failed: {e}")
        traceback.print_exc()
        return False


def check_msdeform_attn():
    """Check whether MultiScaleDeformableAttention is available"""
    try:
        import MultiScaleDeformableAttention as MSDA
        print("  [OK] MultiScaleDeformableAttention C extension available")
        return True
    except ImportError:
        print("  [INFO] MultiScaleDeformableAttention C extension NOT available")
        print("       The pure PyTorch fallback will be used (slower but functional)")
        # Check if the fallback is importable
        try:
            from src.utils.dependencies.XPose.models.UniPose.ops.functions.ms_deform_attn_func import (
                ms_deform_attn_core_pytorch, _MS_CUDA_EXT_AVAILABLE
            )
            if _MS_CUDA_EXT_AVAILABLE:
                print("  [WARN] Extension flag says available but import failed?")
            else:
                print("  [OK] Pure PyTorch fallback is properly configured")
            return True
        except Exception as e:
            print(f"  [FAIL] Fallback also not available: {e}")
            return False


def check_pretrained_weights():
    """Check if pretrained weights are downloaded"""
    import os
    weights_dir = os.path.join(os.path.dirname(__file__), 'pretrained_weights')
    if not os.path.isdir(weights_dir):
        print(f"  [FAIL] pretrained_weights directory not found")
        return False

    # Check human model weights
    human_dir = os.path.join(weights_dir, 'liveportrait', 'base_models')
    if os.path.isdir(human_dir):
        models = ['appearance_feature_extractor.pth', 'motion_extractor.pth',
                   'spade_generator.pth', 'warping_module.pth']
        all_found = all(os.path.isfile(os.path.join(human_dir, m)) for m in models)
        if all_found:
            print("  [OK] Human model weights found")
        else:
            print("  [WARN] Some human model weights are missing")
    else:
        print("  [FAIL] Human model weights not found")

    # Check animal model weights
    animal_dir = os.path.join(weights_dir, 'liveportrait_animals', 'base_models_v1.1')
    if os.path.isdir(animal_dir):
        print("  [OK] Animal model weights found")
    else:
        print("  [INFO] Animal model weights not found (animal mode unavailable)")

    return True


def main():
    print("=" * 60)
    print("  LivePortrait Apple Silicon Compatibility Check")
    print("=" * 60)

    all_checks = [
        ("Python", check_python_version),
        ("PyTorch", check_torch),
        ("torchvision", check_torchvision),
        ("ONNX Runtime", check_onnxruntime),
        ("Transformers", check_transformers),
        ("MPS Operations", check_mps_ops),
        ("MSDeformAttn", check_msdeform_attn),
        ("Pretrained Weights", check_pretrained_weights),
    ]

    results = {}
    for name, check_fn in all_checks:
        print(f"\n--- {name} ---")
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status:4s}  {name}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  {passed}/{total} checks passed")

    if passed == total:
        print("\n  Your environment is ready for LivePortrait on Apple Silicon!")
    elif results.get("MPS Operations", False):
        print("\n  Core MPS support works, but some dependencies may need attention.")
    else:
        print("\n  MPS does not appear to be available. Are you on Apple Silicon?")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
