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

    config = configs.RootConfig(pipeline=secs.PipelineConfig(name=command))
    result = executor.execute_pipeline(config)

    assert len(called) == 1
    assert called[0] is config
    if command == 'study-sweep':
        assert result == 'workflow_run_result'


def test_execute_pipeline_unknown_command_raises_key_error():
    '''
    Given: A RootConfig with an unrecognized pipeline command.
    When: `execute_pipeline` is called.
    Then: Raise a KeyError.
    '''
    config = configs.RootConfig(
        pipeline=secs.PipelineConfig(name='non-existent-command')
    )
    with pytest.raises(KeyError, match='Unknown command'):
        executor.execute_pipeline(config)
