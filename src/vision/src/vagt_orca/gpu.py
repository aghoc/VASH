from __future__ import annotations


class GPUUnavailableError(RuntimeError):
    pass


def require_cuda(device: int = 0) -> str:
    """Return an Ultralytics CUDA device index or fail without CPU fallback."""
    try:
        import torch
    except ImportError as exc:
        raise GPUUnavailableError("PyTorch is not installed; create the uv environment first") from exc
    if not torch.cuda.is_available():
        raise GPUUnavailableError(
            "CUDA is unavailable. CPU fallback is disabled. Check the NVIDIA driver, "
            "/dev/nvidia* device exposure, and the CUDA PyTorch build."
        )
    count = torch.cuda.device_count()
    if device < 0 or device >= count:
        raise GPUUnavailableError(f"CUDA device {device} does not exist (device_count={count})")
    torch.cuda.set_device(device)
    return str(device)


def gpu_summary(device: int = 0) -> dict[str, str | int | float]:
    import torch

    require_cuda(device)
    props = torch.cuda.get_device_properties(device)
    return {
        "device": device,
        "name": props.name,
        "total_memory_gib": round(props.total_memory / 1024**3, 2),
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
    }


def prepare_onnxruntime_cuda() -> list[str]:
    """Preload CUDA/cuDNN site packages and reject ONNX Runtime CPU fallback."""
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        # Empty directory preference skips an incompatible CUDA-major PyTorch install and uses
        # the CUDA/cuDNN packages locked for ONNX Runtime first.
        ort.preload_dlls(directory="")
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise GPUUnavailableError(
            f"ONNX Runtime CUDAExecutionProvider is unavailable; providers={providers}"
        )
    return providers
