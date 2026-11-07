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

'''Unit test suite for device resolution and querying (device.py).'''

# third-party imports
import pytest
import torch
# local imports
import landseg.utils.device as device_utils


def test_resolve_device_explicit_cpu():
    '''
    Given: An explicit 'cpu' target string.
    When: Calling resolve_device.
    Then: Return 'cpu'.
    '''
    assert device_utils.resolve_device('cpu') == 'cpu'


def test_resolve_device_torch_device():
    '''
    Given: A torch.device instance.
    When: Calling resolve_device.
    Then: Return string representation of the device.
    '''
    dev = torch.device('cpu')
    assert device_utils.resolve_device(dev) == 'cpu'


def test_resolve_device_auto_cpu_fallback(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: 'auto' target with both CUDA and MPS disabled.
    When: Calling resolve_device.
    Then: Fall back to 'cpu'.
    '''
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    if hasattr(torch.backends, 'mps'):
        monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)

    assert device_utils.resolve_device('auto') == 'cpu'


def test_resolve_device_auto_cuda(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: 'auto' target with CUDA available.
    When: Calling resolve_device.
    Then: Return 'cuda'.
    '''
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    assert device_utils.resolve_device('auto') == 'cuda'


def test_resolve_device_cuda_unavailable_raises(
    monkeypatch: pytest.MonkeyPatch
):
    '''
    Given: Explicit 'cuda' target when CUDA is not available.
    When: Calling resolve_device.
    Then: Raise ValueError with descriptive message.
    '''
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ValueError, match='CUDA is not available'):
        device_utils.resolve_device('cuda')


def test_get_device_name_cpu():
    '''
    Given: Target 'cpu' device.
    When: Calling get_device_name.
    Then: Return 'cpu'.
    '''
    assert device_utils.get_device_name('cpu') == 'cpu'


def test_get_device_name_mps():
    '''
    Given: Target 'mps' device.
    When: Calling get_device_name.
    Then: Return 'Apple Silicon (MPS)'.
    '''
    assert device_utils.get_device_name('mps') == 'Apple Silicon (MPS)'


def test_get_device_name_cuda(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: Target 'cuda' device with mocked device name.
    When: Calling get_device_name.
    Then: Return mocked GPU device name.
    '''
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(
        torch.cuda, 'get_device_name', lambda _: 'NVIDIA RTX 4090'
    )
    assert device_utils.get_device_name('cuda:0') == 'NVIDIA RTX 4090'

