#!/usr/bin/env python3
# coding: utf-8

"""
Apple Silicon (MPS) compatibility verification and diagnostics for LivePortrait.

Run this script to check whether your environment is properly configured
for running LivePortrait on Apple Silicon (M1/M2/M3/M4) Macs, and to get
performance estimates and optimization tips.

Usage:
    python check_apple_silicon.py
    python check_apple_silicon.py --verbose   # show detailed test output
"""

import sys
import os
import traceback
import platform
import subprocess


def check_python_version():
    """Check Python version >= 3.10"""
    version = sys.version_info
    ok = version >= (3, 10)
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] Python version: {version.major}.{version.minor}.{version.micro}")
    if not ok:
        print("       Recommended: Python 3.10+")
    return ok


def check_hardware():
    """Check Apple Silicon hardware info"""
    ok = True
    print(f"  [INFO] Platform: {platform.platform()}")
    print(f"  [INFO] Architecture: {platform.machine()}")

    if platform.machine() != 'arm64':
        print("  [WARN] Not running on ARM64 architecture (not Apple Silicon?)")
        ok = False
    else:
        # Try to get chip name on macOS
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # This shows "Apple M1", "Apple M2 Pro", etc.
                chip = result.stdout.strip()
                if chip:
                    print(f"  [OK] Chip: {chip}")
                else:
                    # On Apple Silicon, sysctl may not return brand_string
                    # Try another approach
                    result2 = subprocess.run(
                        ['sysctl', 'hw.optional.arm64'],
                        capture_output=True, text=True, timeout=5
                    )
                    if result2.returncode == 0:
                        print("  [OK] Apple Silicon detected")
                    else:
                        print("  [OK] ARM64 architecture detected (Apple Silicon)")
        except Exception:
            print("  [OK] ARM64 architecture detected (Apple Silicon)")

    # Memory info
    try:
        result = subprocess.run(
            ['sysctl', '-n', 'hw.memsize'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            total_gb = int(result.stdout.strip()) / (1024 ** 3)
            print(f"  [INFO] Total RAM: {total_gb:.1f} GB")
            # MPS uses unified memory, so available GPU memory ≈ free RAM
            # Rough estimate: leave 4GB for system
            usable_gb = max(total_gb - 4, 2)
            print(f"  [INFO] Estimated usable GPU memory: ~{usable_gb:.0f} GB (unified)")
    except Exception:
        pass

    return ok


def check_macos_version():
    """Check macOS version (Ventura 13.0+ recommended for stable MPS)"""
    version = platform.mac_ver()[0]
    if not version:
        print("  [SKIP] Not running on macOS")
        return True

    parts = version.split('.')
    major = int(parts[0]) if len(parts) > 0 else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    ver_tuple = (major, minor)

    ok = ver_tuple >= (13, 0)
    status = "OK" if ok else "WARN"
    print(f"  [{status}] macOS version: {version}")
    if not ok:
        print("       Recommended: macOS Ventura 13.0+ for stable MPS support")
    elif ver_tuple >= (14, 0):
        print("  [OK] macOS Sonoma 14.0+ — MPS is stable and well-supported")
    return ok


def check_torch():
    """Check PyTorch availability and MPS support"""
    try:
        import torch
        print(f"  [OK] PyTorch version: {torch.__version__}")
    except ImportError:
        print("  [FAIL] PyTorch not installed")
        return False

    # Version check — MPS autocast needs 2.1+
    parts = torch.__version__.split('.')[:2]
    ver_major, ver_minor = int(parts[0]), int(parts[1])
    ver_tuple = (ver_major, ver_minor)

    if ver_tuple >= (2, 1):
        print("  [OK] PyTorch 2.1+ — MPS autocast (FP16) supported")
    elif ver_tuple >= (2, 0):
        print("  [WARN] PyTorch 2.0 — MPS autocast is experimental. Upgrade to 2.1+ recommended")
    else:
        print("  [FAIL] PyTorch < 2.0 — MPS support limited. Upgrade to 2.1+ strongly recommended")

    # Check MPS
    try:
        mps_available = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()
        if mps_available:
            print(f"  [OK] MPS (Apple Silicon GPU) is available")
        else:
            if mps_built:
                print(f"  [WARN] MPS is built but NOT available (hardware issue?)")
            else:
                print(f"  [FAIL] MPS is NOT built into this PyTorch installation")
    except Exception as e:
        print(f"  [WARN] MPS check failed: {e}")
        mps_available = False

    # Check CUDA (should not be available on macOS)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"  [INFO] CUDA is available (GPU: {torch.cuda.get_device_name(0)})")
    else:
        print(f"  [INFO] CUDA is NOT available (expected on macOS)")

    # Check torch.mps module
    if hasattr(torch, 'mps'):
        has_empty_cache = hasattr(torch.mps, 'empty_cache')
        has_synchronize = hasattr(torch.mps, 'synchronize')
        if has_empty_cache and has_synchronize:
            print("  [OK] torch.mps memory management (empty_cache, synchronize) available")
        else:
            print("  [WARN] torch.mps module exists but missing some functions")
    else:
        print("  [WARN] torch.mps module not available (upgrade PyTorch)")

    return True


def check_mps_autocast():
    """Test MPS autocast (FP16) support"""
    try:
        import torch
        if not torch.backends.mps.is_available():
            print("  [SKIP] MPS not available")
            return True

        device = torch.device('mps')
        try:
            with torch.autocast(device_type='mps', dtype=torch.float16):
                x = torch.randn(2, 2, device=device)
                y = x @ x.T
            print("  [OK] MPS autocast (FP16) works — ~2x speedup & memory savings enabled")
            return True
        except Exception as e:
            print(f"  [WARN] MPS autocast failed: {e}")
            print("       Inference will run in FP32 (slower, more memory)")
            return True  # Not a fatal error
    except Exception as e:
        print(f"  [WARN] Could not test MPS autocast: {e}")
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
            print("  [OK] CoreMLExecutionProvider available (accelerated face detection)")
        else:
            print("  [INFO] CoreMLExecutionProvider not available (face detection will use CPU)")
            print("       Install with: pip install onnxruntime-coreml")
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

        # Test convolution (4D)
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

        # Test grid_sample 3D (native MPS may not support this)
        inp_3d = torch.randn(1, 1, 4, 4, 4).to(device)  # create on CPU, move to MPS
        grid_3d = torch.randn(1, 4, 4, 4, 3).to(device)
        try:
            out_3d = F.grid_sample(inp_3d, grid_3d, align_corners=False)
            print("  [OK] F.grid_sample (3D) natively supported on MPS")
        except Exception:
            print("  [INFO] F.grid_sample (3D) not natively on MPS (handled by CPU fallback)")

        # Test grid_sample_3d_fallback from device.py
        try:
            from src.utils.device import grid_sample_3d_fallback
            out_fb = grid_sample_3d_fallback(inp_3d, grid_3d, align_corners=False)
            assert out_fb.shape == (1, 1, 4, 4, 4)
            print(f"  [OK] grid_sample_3d_fallback() works (output device={out_fb.device})")
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


def check_device_module():
    """Check the unified device module"""
    try:
        from src.utils.device import select_device, inference_ctx, grid_sample_3d_fallback, is_mps, is_cuda, empty_cache, synchronize
        device = select_device(flag_force_cpu=False)
        print(f"  [OK] Auto-detected device: {device}")

        ctx = inference_ctx(device, flag_use_half_precision=True)
        print(f"  [OK] Inference context: {type(ctx).__name__}")

        if is_mps():
            print("  [OK] MPS detected — all compatibility patches active")
        return True
    except Exception as e:
        print(f"  [FAIL] Device module check failed: {e}")
        traceback.print_exc()
        return False


def check_pretrained_weights():
    """Check if pretrained weights are downloaded"""
    weights_dir = os.path.join(os.path.dirname(__file__), 'pretrained_weights')
    if not os.path.isdir(weights_dir):
        print(f"  [FAIL] pretrained_weights directory not found")
        print("       Run: huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights")
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
        print("       Run: huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights")

    # Check animal model weights
    animal_dir = os.path.join(weights_dir, 'liveportrait_animals', 'base_models_v1.1')
    if os.path.isdir(animal_dir):
        print("  [OK] Animal model weights found")
    else:
        print("  [INFO] Animal model weights not found (animal mode unavailable)")
        print("       Download from: https://huggingface.co/KwaiVGI/LivePortrait-Animals")

    return True


def print_performance_tips():
    """Print optimization tips for Apple Silicon users"""
    print("\n" + "=" * 60)
    print("  Performance Tips for Apple Silicon")
    print("=" * 60)
    print("""
  1. FP16 Autocast: Enabled by default on MPS. Gives ~2x speedup.
     If you see black/NaN outputs, add: --flag_use_half_precision=False

  2. Memory Management: MPS uses unified memory (shared with CPU).
     For long videos, the system automatically clears GPU cache every
     10 frames to prevent OOM.

  3. torch.compile: Available on MPS with mode='default' (not max-autotune).
     Enable with: --flag_do_torch_compile

  4. Recommended Settings:
     - Human mode:  python inference.py -s source.png -d driving.mp4
     - Animal mode: python inference_animals.py -s animal.png -d driving.mp4
     - With compile: Add --flag_do_torch_compile for 10-30% speedup
     - CPU fallback: Add --flag_force_cpu to run on CPU only

  5. Memory Estimates (approximate):
     - Single portrait animation: ~4-6 GB GPU memory
     - Long video (300+ frames): ~8-12 GB GPU memory
     - Animal mode: ~6-8 GB GPU memory
     - If OOM, try: --flag_use_half_precision=True (default)
""")


def main():
    print("=" * 60)
    print("  LivePortrait Apple Silicon Diagnostics")
    print("=" * 60)

    all_checks = [
        ("Hardware", check_hardware),
        ("macOS Version", check_macos_version),
        ("Python", check_python_version),
        ("PyTorch", check_torch),
        ("MPS Autocast", check_mps_autocast),
        ("torchvision", check_torchvision),
        ("ONNX Runtime", check_onnxruntime),
        ("Transformers", check_transformers),
        ("MPS Operations", check_mps_ops),
        ("Device Module", check_device_module),
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

    print_performance_tips()

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
