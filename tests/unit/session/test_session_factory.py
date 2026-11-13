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

'''Unit tests for session factory (session/factory.py).'''

# standard imports
import unittest.mock
# local imports
import landseg.session.engine.epoch as epoch_mod
import landseg.session.factory as factory_mod
import landseg.session.instrumentation.dashboards as dashboards
import landseg.session.orchestration.runner as runner_mod


# ----- `session/factory.py` builder entry point tests
def test_build_session_runner_overfit(
    session_config,
    dataspecs,
    mock_model,
    mock_session_paths,
):
    '''
    Given: Data specs, mock model, session config, and logger.
    When: Calling `build_session_runner` for overfit.
    Then: Return `EpochEngine` configured in 'train_eval' mode.
    '''
    engine_session = factory_mod.build_session_runner(
        dataspecs=dataspecs,
        model=mock_model,
        config=session_config,
        context=_get_context(mock_session_paths),
        session_type='overfit',
    )

    assert isinstance(engine_session, epoch_mod.EpochRunner)
    assert engine_session.mode == 'train_eval'


def test_build_session_runner_evaluate(
    session_config,
    dataspecs,
    mock_model,
    mock_session_paths,
):
    '''
    Given: Data specs, mock model, session config, and logger.
    When: Calling `build_session_runner` for evaluation.
    Then: Return `EpochEngine` configured in 'eval_only' mode.
    '''
    engine_session = factory_mod.build_session_runner(
        dataspecs=dataspecs,
        model=mock_model,
        config=session_config,
        context=_get_context(mock_session_paths),
        session_type='evaluate',
    )

    assert isinstance(engine_session, epoch_mod.EpochRunner)
    assert engine_session.mode == 'eval_only'


def test_build_session_runner_continuous(
    session_config,
    dataspecs,
    mock_model,
    mock_session_paths,
):
    '''
    Given: Valid session context with results paths manager.
    When: Calling `build_session_runner` for continuous training.
    Then: Return `ContinuousRunner` orchestrator.
    '''
    runner = factory_mod.build_session_runner(
        dataspecs=dataspecs,
        model=mock_model,
        config=session_config,
        context=_get_context(mock_session_paths),
        session_type='continuous',
    )

    assert isinstance(runner, runner_mod.ContinuousRunner)


def test_build_session_runner_curriculum(
    session_config,
    dataspecs,
    mock_model,
    mock_session_paths,
):
    '''
    Given: Valid session context with results paths manager.
    When: Calling `build_session_runner` for curriculum training.
    Then: Return `CurriculumRunner` orchestrator.
    '''
    session_config.orchestration.curriculum.schema = 'baseline'
    runner = factory_mod.build_session_runner(
        dataspecs=dataspecs,
        model=mock_model,
        config=session_config,
        context=_get_context(mock_session_paths),
        session_type='curriculum',
    )

    assert isinstance(runner, runner_mod.CurriculumRunner)


def test_build_session_runner_databricks_tracker_defaults_to_mlflow(
    session_config,
    dataspecs,
    mock_model,
    mock_session_paths,
):
    '''
    Given: SessionBuildContext with is_databricks=True and without explicit trackers.
    When: Calling `build_session_runner`.
    Then: Automatically defaults to MLflow tracker with 'databricks' URI.
    '''
    with unittest.mock.patch('mlflow.set_tracking_uri') as mock_set_uri, \
         unittest.mock.patch('mlflow.start_run'):
        runner = factory_mod.build_session_runner(
            dataspecs=dataspecs,
            model=mock_model,
            config=session_config,
            context=_get_context(mock_session_paths, is_databricks=True),
            session_type='continuous',
        )
        assert isinstance(runner, runner_mod.ContinuousRunner)
        mock_set_uri.assert_called_with('databricks')


def test_base_tracker_string_uri():
    '''
    Given: BaseTracker or MLFlowTracker with a string URI (e.g. 'databricks').
    When: Initializing tracker.
    Then: Safely stores the string URI without calling .absolute().
    '''
    with unittest.mock.patch('mlflow.set_tracking_uri'), \
         unittest.mock.patch('mlflow.start_run'):
        tracker = dashboards.MLFlowTracker('databricks')
        assert tracker.uri == 'databricks'


# ----- internal helpers
def _get_context(mock_session_paths, **kwargs):
    return factory_mod.SessionBuildContext(
        device='cpu',
        session_paths=mock_session_paths,
        eval_dataset='val',
        **kwargs,
    )
