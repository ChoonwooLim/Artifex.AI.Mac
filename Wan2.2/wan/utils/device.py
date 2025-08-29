import torch
from contextlib import nullcontext


def get_best_device(preferred_cuda_index: int = 0) -> torch.device:
    """Return the best available device: CUDA -> MPS -> CPU.

    On macOS/Apple Silicon, MPS is preferred when CUDA is not available.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{preferred_cuda_index}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_effective_param_dtype(requested: torch.dtype, device: torch.device) -> torch.dtype:
    """Choose a safe/fast dtype for the target device.

    - MPS: prefer float16; downcast bf16->fp16 if requested is bf16
    - CPU: prefer float32 for numerical stability if low-precision requested
    - CUDA: keep requested
    """
    if device.type == "mps":
        if requested == torch.bfloat16:
            return torch.float16
        return torch.float16 if requested == torch.float16 else requested
    if device.type == "cpu":
        # Avoid fp16/bf16 on CPU unless explicitly needed
        if requested in (torch.float16, torch.bfloat16):
            return torch.float32
        return requested
    return requested


def autocast(device: torch.device, dtype: torch.dtype):
    """Autocast context depending on device.

    - CUDA: use cuda autocast with requested dtype
    - CPU: use cpu autocast with bf16 when possible (fallback to no autocast)
    - MPS: no autocast (use nullcontext)
    """
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=dtype)
    if device.type == "cpu":
        # Prefer bf16 on CPU if user requested low-precision, else no autocast
        try:
            target = dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
            return torch.amp.autocast("cpu", dtype=target)
        except Exception:
            return nullcontext()
    return nullcontext()


def empty_cache_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


