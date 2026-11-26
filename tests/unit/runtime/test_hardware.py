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

'''Unit tests for device resolution and hardware queries.'''

# third-party imports
import pytest
import torch
# local imports
import landseg.runtime.hardware as hardware


def test_resolve_device_explicit_cpu():
    '''
    Given: An explicit 'cpu' target string.
    When: Calling resolve_device.
    Then: Return 'cpu'.
    '''
    assert hardware.resolve_device('cpu') == 'cpu'


def test_resolve_device_torch_device():
    '''
    Given: A torch.device instance.
    When: Calling resolve_device.
    Then: Return string representation of the device.
    '''
    dev = torch.device('cpu')
    assert hardware.resolve_device(dev) == 'cpu'


def test_resolve_device_auto_cpu_fallback(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: 'auto' target with CUDA and MPS unavailable.
    When: Calling resolve_device.
    Then: Return 'cpu'.
    '''
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    if hasattr(torch.backends, 'mps'):
        monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)

    assert hardware.resolve_device('auto') == 'cpu'


def test_resolve_device_auto_cuda(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: 'auto' target with CUDA available.
    When: Calling resolve_device.
    Then: Return 'cuda'.
    '''
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)

    assert hardware.resolve_device('auto') == 'cuda'


def test_resolve_device_cuda_unavailable_raises(
    monkeypatch: pytest.MonkeyPatch
):
    '''
    Given: Explicit 'cuda' target requested but CUDA is unavailable.
    When: Calling resolve_device.
    Then: Raise ValueError.
    '''
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    with pytest.raises(ValueError, match='CUDA is not available'):
        hardware.resolve_device('cuda')


def test_get_device_name_cpu():
    '''
    Given: CPU device identifier.
    When: Calling get_device_name.
    Then: Return 'cpu'.
    '''
    assert hardware.get_device_name('cpu') == 'cpu'


def test_get_device_name_mps():
    '''
    Given: MPS device identifier.
    When: Calling get_device_name.
    Then: Return 'Apple Silicon (MPS)'.
    '''
    assert hardware.get_device_name('mps') == 'Apple Silicon (MPS)'


def test_get_device_name_cuda(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: CUDA device identifier with CUDA available and mocked name.
    When: Calling get_device_name.
    Then: Return the mocked device name.
    '''
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(
        torch.cuda, 'get_device_name', lambda idx: 'NVIDIA RTX 4090'
    )
    assert hardware.get_device_name('cuda:0') == 'NVIDIA RTX 4090'


def test_is_cpu_bf16_supported(mocker):
    '''
    Given: PyTorch internal CPU feature probe functions.
    When: Testing is_cpu_bf16_supported helper.
    Then: Accurately reflect AVX-512 BF16 and AMX capability.
    '''
    mock_cpu = mocker.MagicMock()
    mock_cpu._is_avx512_bf16_supported.return_value = True
    mocker.patch.object(torch._C, '_cpu', mock_cpu, create=True)
    assert hardware.is_cpu_bf16_supported() is True

    mock_cpu._is_avx512_bf16_supported.return_value = False
    mock_cpu._is_amx_tile_supported.return_value = True
    assert hardware.is_cpu_bf16_supported() is True

    mock_cpu._is_amx_tile_supported.return_value = False
    assert hardware.is_cpu_bf16_supported() is False


@pytest.mark.parametrize(
    'bf16_supported, expected',
    [(True, torch.bfloat16), (False, None)],
)
def test_resolve_amp_dtype_cpu(
    monkeypatch: pytest.MonkeyPatch,
    bf16_supported: bool,
    expected: torch.dtype | None,
):
    '''
    Given: CPU device with or without hardware BF16.
    When: Calling resolve_amp_dtype.
    Then: Return bfloat16 when supported, otherwise None (FP32).
    '''
    monkeypatch.setattr(
        hardware, 'is_cpu_bf16_supported', lambda: bf16_supported
    )
    assert hardware.resolve_amp_dtype('cpu') is expected


@pytest.mark.parametrize(
    'name, capability, expected',
    [
        ('NVIDIA T600 Laptop GPU', (7, 5), None),
        ('NVIDIA GeForce GTX 1650', (7, 5), None),
        ('NVIDIA GeForce GTX 1080', (6, 1), None),
        ('NVIDIA GeForce RTX 2080', (7, 5), torch.float16),
        ('NVIDIA RTX 4090', (8, 9), torch.float16),
    ],
)
def test_resolve_amp_dtype_cuda(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    capability: tuple[int, int],
    expected: torch.dtype | None,
):
    '''
    Given: CUDA device with a given name and compute capability.
    When: Calling resolve_amp_dtype.
    Then: Return float16 only for GPUs with Tensor Cores, else None.
    '''
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda idx: name)
    monkeypatch.setattr(
        torch.cuda, 'get_device_capability', lambda idx: capability
    )
    assert hardware.resolve_amp_dtype('cuda:0') is expected


def test_resolve_amp_dtype_mps():
    '''
    Given: MPS device identifier.
    When: Calling resolve_amp_dtype.
    Then: Return float16 (unchanged behaviour).
    '''
    assert hardware.resolve_amp_dtype('mps') is torch.float16
