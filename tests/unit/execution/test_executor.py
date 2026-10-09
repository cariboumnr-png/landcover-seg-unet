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

'''Unit tests for pipeline execution dispatching logic.'''

# third-party imports
import pytest
# local imports
import landseg.configs as configs
import landseg.configs.schema.sections as secs
import landseg.execution.executor as executor
import landseg.execution.pipelines as pipelines
import landseg.execution.workflows as workflows


# ----- `execute_pipeline` tests
@pytest.mark.parametrize('command,target_mod,target_attr,is_class', [
    ('default', workflows, 'execute_default_action', False),
    ('world-grid', pipelines, 'WorldGridGeneration', True),
    ('data-harmonize', pipelines, 'DataHarmonization', True),
    ('data-ingest', pipelines, 'DataIngestion', True),
    ('data-prepare', pipelines, 'DataPreparation', True),
    ('model-train', pipelines, 'ModelTraining', True),
    ('model-evaluate', pipelines, 'ModelEvaluation', True),
    ('diagnose-overfit', workflows, 'execute_diagnose_overfit', False),
    ('batch-ingest', workflows, 'execute_batch_ingest', False),
    ('e2e-intake', workflows, 'execute_e2e_intake', False),
    ('e2e-experiment', workflows, 'execute_e2e_experiment', False),
    ('study-analysis', workflows, 'execute_study_analysis', False),
    ('study-sweep', workflows, 'execute_study_sweep', False),
])
def test_execute_pipeline_dispatch(
    monkeypatch,
    command: str,
    target_mod,
    target_attr: str,
    is_class: bool,
):
    '''
    Given: A RootConfig with a registered pipeline command.
    When: `execute_pipeline` is called.
    Then: Dispatch correctly to target pipeline or workflow callable.
    '''
    called = []

    if is_class:
        class MockPipeline:
            def __init__(self, cfg):
                self.cfg = cfg

            def run(self):
                called.append(self.cfg)
                return 'pipeline_run_result'

        monkeypatch.setattr(target_mod, target_attr, MockPipeline)
    else:
        def mock_workflow(cfg):
            called.append(cfg)
            return 'workflow_run_result'

        monkeypatch.setattr(target_mod, target_attr, mock_workflow)

    config = configs.RootConfig(command=secs.CommandConfig(name=command))
    result = executor.execute_pipeline(config)

    assert len(called) == 1
    assert called[0] is config
    if command == 'study-sweep':
        assert result == 'workflow_run_result'


def test_execute_pipeline_preflight_dispatch(mocker):
    '''
    Given: A RootConfig with command='preflight'.
    When: `execute_pipeline` is called.
    Then: Dispatch to `preflight.run_preflight`.
    '''
    mock_run = mocker.patch(
        'landseg.execution.preflight.run_preflight',
        return_value='preflight_mock_result'
    )
    config = configs.RootConfig(
        command=secs.CommandConfig(name='preflight'),
        execution=configs.ExecutionContext(preflight_target='model-train'),
    )
    result = executor.execute_pipeline(config)
    mock_run.assert_called_once_with(config)
    assert result == 'preflight_mock_result'


def test_execute_pipeline_preflight_invalid_target_raises_key_error():
    '''
    Given: A RootConfig with command='preflight' and unknown target.
    When: `execute_pipeline` is called.
    Then: Raise a KeyError from the preflight engine.
    '''
    config = configs.RootConfig(
        command=secs.CommandConfig(name='preflight'),
        execution=configs.ExecutionContext(preflight_target='unknown-target'),
    )
    with pytest.raises(KeyError, match='not supported for preflight'):
        executor.execute_pipeline(config)


def test_execute_pipeline_unknown_command_raises_key_error():
    '''
    Given: A RootConfig with an unrecognized pipeline command.
    When: `execute_pipeline` is called.
    Then: Raise a KeyError.
    '''
    config = configs.RootConfig(
        command=secs.CommandConfig(name='non-existent-command')
    )
    with pytest.raises(KeyError, match='Unknown command'):
        executor.execute_pipeline(config)

