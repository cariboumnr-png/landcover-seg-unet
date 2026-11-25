# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      © King's Printer for Ontario, 2026.                    #
#                                                                             #
#       Licensed under the Apache License, Version 2.0 (the 'License');       #
#          you may not use this file except in compliance with the            #
#                                  License.                                   #
#                  You may obtain a copy of the License at:                   #
#                                                                             #
#                  http://www.apache.org/licenses/LICENSE-2.0                 #
#                                                                             #
#    Unless required by applicable law or agreed to in writing, software      #
#     distributed under the License is distributed on an 'AS IS' BASIS,       #
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        #
#                                   implied.                                  #
#       See the License for the specific language governing permissions       #
#                       and limitations under the License.                    #
# =========================================================================== #

'''
Hardware device resolution, query, and capability probing utilities.

Provides safe runtime hardware resolution for execution devices,
supporting automatic probe, CUDA, Apple Silicon MPS, and CPU fallback,
along with microarchitecture capability probes for BF16 and cuDNN.

Public APIs:
    - `get_device_name`: Query human-readable name of execution device.
    - `is_cpu_bf16_supported`: Probe whether CPU supports hardware BF16.
    - `resolve_amp_dtype`: Resolve the safe autocast dtype for a device.
    - `resolve_device`: Resolve hardware execution target string.
'''

# standard imports
import re
# third-party imports
import torch

# turing TU116/TU117 GPUs report cc 7.5 but lack Tensor Cores (#88)
_NO_TENSOR_CORE_GPUS = re.compile(
    r'GTX 16\d\d|\bT(?:400|600|1000)\b|\bMX(?:450|550)\b'
)


# ----- public functions
def resolve_device(target: str | torch.device = 'auto') -> str:
    '''
    Resolve target device string into a concrete hardware device.

    Evaluates execution hardware availability when 'auto' is specified,
    prioritizing CUDA, then Apple Silicon MPS, and falling back to CPU.

    Args:
        target:
            Target device specification string (e.g., 'auto', 'cpu',
            'cuda', 'cuda:0', 'mps') or a torch.device instance.

    Returns:
        str:
            Resolved concrete device identifier string.

    Raises:
        ValueError:
            If a specific CUDA device is requested but unavailable.
    '''
    if isinstance(target, torch.device):
        return str(target)

    target_clean = target.strip().lower()
    if target_clean == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    if target_clean.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError(
            f"Device '{target}' requested, but CUDA is not available."
        )

    return target_clean


def get_device_name(device: str | torch.device = 'auto') -> str:
    '''
    Return human-readable device name for the specified device.

    Resolves device metadata safely without raising exceptions if the
    specified device is unavailable.

    Args:
        device:
            Target device identifier or torch device object.

    Returns:
        str:
            Human-readable hardware description or device identifier.
    '''
    resolved = resolve_device(device) if device == 'auto' else str(device)
    dev = torch.device(resolved)

    if dev.type == 'cuda' and torch.cuda.is_available():
        idx = dev.index if dev.index is not None else 0
        try:
            return torch.cuda.get_device_name(idx)
        except Exception:  # pylint: disable=broad-exception-caught
            return f'cuda:{idx}'

    if dev.type == 'mps':
        return 'Apple Silicon (MPS)'

    return dev.type


def resolve_amp_dtype(device: str | torch.device) -> torch.dtype | None:
    '''
    Resolve the autocast dtype that is safe on the given device.

    CPUs use BF16 only with hardware acceleration. CUDA devices use
    FP16 only when they have Tensor Cores (compute capability >= 7.0
    and not a TU116/TU117 part), since cuDNN FP16 convolutions emit
    `NaN` on those GPUs (Issue #88). Other devices keep FP16.

    Args:
        device:
            Concrete device identifier (e.g., 'cpu', 'cuda', 'cuda:0',
            'mps') or a torch.device instance.

    Returns:
        torch.dtype | None:
            Autocast dtype, or None when the device should run FP32.
    '''
    dev = torch.device(device)
    if dev.type == 'cpu':
        return torch.bfloat16 if is_cpu_bf16_supported() else None
    if dev.type == 'cuda':
        idx = dev.index if dev.index is not None else 0
        major, _ = torch.cuda.get_device_capability(idx)
        name = torch.cuda.get_device_name(idx)
        if major < 7 or _NO_TENSOR_CORE_GPUS.search(name):
            return None
        return torch.float16
    return torch.float16


def is_cpu_bf16_supported() -> bool:
    '''
    Probe whether host CPU supports hardware-accelerated BF16.

    Hardware BF16 on x86_64 requires AVX-512 (AVX512BF16) or AMX-BF16.
    CPUs lacking these instructions incur substantial software emulation
    penalties or numerical divergence when running under bfloat16.

    Returns:
        bool:
            True if CPU natively accelerates BF16 arithmetic.
    '''
    cpu_mod = getattr(getattr(torch, '_C', None), '_cpu', None)
    if cpu_mod is not None:
        avx512 = getattr(cpu_mod, '_is_avx512_bf16_supported', None)
        if callable(avx512) and avx512():
            return True
        amx = getattr(cpu_mod, '_is_amx_tile_supported', None)
        if callable(amx) and amx():
            return True
    return False
