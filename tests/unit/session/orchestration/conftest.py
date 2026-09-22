# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      (c) King's Printer for Ontario, 2026.                  #
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
# pylint: disable=too-few-public-methods

'''Fixtures for testing `landseg.session.orchestration` subpackage.'''

# third-party imports
import pytest
# local imports
import landseg.session.engine.epoch.runner as epoch_executor
import landseg.session.orchestration.runner as runner_mod


# ----- pytest fixtures
@pytest.fixture
def mock_epoch_engine(mock_trainer, mock_evaluator):
    return epoch_executor.EpochRunner(
        mode='train_eval',
        trainer=mock_trainer,
        evaluator=mock_evaluator
    )


@pytest.fixture
def mock_runner_config(mock_session_paths):
    return runner_mod.BaseRunnerConfig(artifacts_paths=mock_session_paths)
