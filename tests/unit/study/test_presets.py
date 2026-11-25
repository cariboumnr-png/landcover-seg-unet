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

# pylint: disable=protected-access

'''Unit tests for `landseg.study.sweep.presets` registry alignment.'''

# third-party imports
import pytest
# local imports
import landseg._constants as c
import landseg.study.sweep.presets as presets
import landseg.study.sweep.presets._registry as registry


# ----- preset registry synchronization tests
def test_sweep_presets_constant_matches_registry():
    '''
    Given: Project-wide `SWEEP_PRESETS` constants and preset `_REGISTRY`.
    When: Comparing declared preset names against registered callables.
    Then: Both sets match exactly and every preset resolves cleanly.
    '''
    assert len(c.SWEEP_PRESETS) == len(set(c.SWEEP_PRESETS))
    assert set(registry._REGISTRY) == set(c.SWEEP_PRESETS)
    assert tuple(registry._REGISTRY) == c.SWEEP_PRESETS

    for preset_name in c.SWEEP_PRESETS:
        resolved = presets.resolve(preset_name)
        assert callable(resolved)


def test_presets_resolve_rejects_unknown_name():
    '''
    Given: An unregistered preset name.
    When: Calling `presets.resolve`.
    Then: Raise ValueError.
    '''
    with pytest.raises(ValueError, match='Unknown preset'):
        presets.resolve('non_existent_preset')
