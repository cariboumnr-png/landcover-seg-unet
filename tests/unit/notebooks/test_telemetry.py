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

'''Unit test suite for `landseg.notebooks.telemetry` interactive module.'''

# third-party imports
import pytest
# local imports
import landseg.notebooks as nb


def test_safe_training_session_normal_exit():
    '''
    Given: An uninterrupted training block wrapped in safe_training_session.
    When: Block executes to completion.
    Then: State records uninterrupted execution.
    '''
    with nb.safe_training_session() as state:
        assert state['interrupted'] is False
        _ = 1 + 1

    assert state['interrupted'] is False


def test_safe_training_session_handles_keyboard_interrupt():
    '''
    Given: A simulated KeyboardInterrupt inside safe_training_session.
    When: Cell is interrupted.
    Then: State records interruption and halts cleanly without crash.
    '''
    with nb.safe_training_session(reraise=False) as state:
        raise KeyboardInterrupt('Simulated notebook stop button click')

    assert state['interrupted'] is True


def test_safe_training_session_reraises_when_requested():
    '''
    Given: safe_training_session with reraise=True.
    When: KeyboardInterrupt occurs.
    Then: Re-raises KeyboardInterrupt after cleanup.
    '''
    with pytest.raises(KeyboardInterrupt):
        with nb.safe_training_session(reraise=True):
            raise KeyboardInterrupt('User stop')
