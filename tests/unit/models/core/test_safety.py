# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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

# pylint: disable=missing-function-docstring
# pylint: disable=protected-access

'''Unit tests for NumericSafety (safety.py).'''

# third-party imports
import torch
# local imports
import landseg.models.core.safety as safety


# ----- `NumericSafety` initialization
def test_numeric_safety_initialization():
    ns = safety.NumericSafety(
        enable_clamp=True, clamp_range=(0.1, 1.0), device='cpu'
    )

    assert ns.enable_clamp is True
    assert ns.clamp_range == (0.1, 1.0)
    assert ns.device == 'cpu'


# ----- `NumericSafety` clamping
def test_numeric_safety_clamp_enabled():
    ns = safety.NumericSafety(
        enable_clamp=True, clamp_range=(0.0, 5.0), device='cpu'
    )
    x = torch.tensor([-1.0, 2.0, 6.0])
    clamped = ns.clamp(x)
    expected = torch.tensor([0.0, 2.0, 5.0])

    assert torch.allclose(clamped, expected)


def test_numeric_safety_clamp_disabled():
    ns = safety.NumericSafety(
        enable_clamp=False, clamp_range=(0.0, 5.0), device='cpu'
    )
    x = torch.tensor([-1.0, 2.0, 6.0])
    clamped = ns.clamp(x)

    assert torch.allclose(clamped, x)


# ----- `NumericSafety` autocast context
def test_numeric_safety_autocast_context():
    ns = safety.NumericSafety(
        enable_clamp=True, clamp_range=(0.0, 5.0), device='cpu'
    )
    ctx = ns.autocast_context(enable=True, dtype=torch.bfloat16)

    assert isinstance(ctx, torch.autocast)
    assert ctx.device == 'cpu'
    assert ctx.fast_dtype == torch.bfloat16
    assert ctx._enabled is True
