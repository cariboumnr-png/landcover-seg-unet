# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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

'''Unit tests for programmatic API module.'''

# third-party imports
import pytest
# local imports
import landseg.adapters.api as api
import landseg.configs as configs


# ----- `api.run` tests
def test_api_run_success(mocker):
    '''
    Given: A valid `RootConfig` instance.
    When: Calling `api.run`.
    Then: Delegate execution to `execution.execute_pipeline`.
    '''
    mock_exec = mocker.patch(
        'landseg.execution.execute_pipeline',
        return_value={'status': 'SUCCESS'}
    )
    cfg = configs.RootConfig()
    cfg.command.name = 'data-harmonize'

    result = api.run(cfg)
    mock_exec.assert_called_once_with(cfg)
    assert result == {'status': 'SUCCESS'}


def test_api_run_keyboard_interrupt(mocker):
    '''
    Given: A pipeline run that is interrupted by user.
    When: `execution.execute_pipeline` raises KeyboardInterrupt.
    Then: Propagate KeyboardInterrupt.
    '''
    mocker.patch(
        'landseg.execution.execute_pipeline',
        side_effect=KeyboardInterrupt
    )
    cfg = configs.RootConfig()

    with pytest.raises(KeyboardInterrupt):
        api.run(cfg)


def test_api_run_exception(mocker):
    '''
    Given: A pipeline run that raises an unhandled exception.
    When: `execution.execute_pipeline` raises RuntimeError.
    Then: Log error and re-raise the exception.
    '''
    mocker.patch(
        'landseg.execution.execute_pipeline',
        side_effect=RuntimeError('Pipeline failed')
    )
    cfg = configs.RootConfig()

    with pytest.raises(RuntimeError, match='Pipeline failed'):
        api.run(cfg)
