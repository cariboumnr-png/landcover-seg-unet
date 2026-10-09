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
    cfg.command = 'data-harmonize'

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


# ----- `api.run_preflight` tests
def test_api_run_preflight_delegates(mocker):
    '''
    Given: A valid `RootConfig` and explicit target string.
    When: Calling `api.run_preflight`.
    Then: Delegate execution to `preflight.run_preflight`.
    '''
    mock_preflight = mocker.patch(
        'landseg.execution.preflight.run_preflight',
        return_value='report'
    )
    cfg = configs.RootConfig()
    cfg.command = 'model-train'

    result = api.run_preflight(cfg, target='model-train', strict=True)
    assert result == 'report'
    assert cfg.execution.preflight_strict_mode is True
    mock_preflight.assert_called_once_with(
        root_config=cfg,
        target='model-train',
        exp_root=None,
    )


def test_api_run_preflight_string_target(mocker):
    '''
    Given: A single string target argument.
    When: Calling `api.run_preflight('model-train')`.
    Then: Initialize default `RootConfig` and pass resolved target.
    '''
    mock_preflight = mocker.patch(
        'landseg.execution.preflight.run_preflight',
        return_value='report'
    )

    result = api.run_preflight('model-train')
    assert result == 'report'
    mock_preflight.assert_called_once()
    assert mock_preflight.call_args.kwargs['target'] == 'model-train'


def test_api_run_preflight_inferred_target(mocker):
    '''
    Given: A `RootConfig` with `command = 'data-harmonize'`.
    When: Calling `api.run_preflight(cfg)` without explicit target.
    Then: Infer target from `command`.
    '''
    mock_preflight = mocker.patch(
        'landseg.execution.preflight.run_preflight',
        return_value='report'
    )
    cfg = configs.RootConfig()
    cfg.command = 'data-harmonize'

    result = api.run_preflight(cfg)
    assert result == 'report'
    mock_preflight.assert_called_once_with(
        root_config=cfg,
        target='data-harmonize',
        exp_root=None,
    )

