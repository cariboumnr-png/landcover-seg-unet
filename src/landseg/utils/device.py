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
Hardware device resolution and query utilities.

Provides safe runtime hardware resolution for execution devices,
supporting automatic probe, CUDA, Apple Silicon MPS, and CPU fallback.

Public APIs:
    - `resolve_device`: Resolve hardware execution target string.
    - `get_device_name`: Query human-readable name of execution device.
'''

# third-party imports
import torch


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
