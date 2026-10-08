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

'''Unit tests for composite e2e workflows.'''

# local imports
import landseg.configs as configs
import landseg.execution.pipelines as pipelines
import landseg.execution.workflows as workflows


# ----- `execute_e2e_intake` tests
def test_execute_e2e_intake_stage_sequence(monkeypatch):
    '''
    Given: A RootConfig instance.
    When: `execute_e2e_intake` is executed.
    Then: Call DataHarmonization then execute_batch_ingest in sequence.
    '''
    calls = []

    class MockHarmonization:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self):
            calls.append('harmonize')

    def mock_batch_ingest(cfg):
        calls.append('batch_ingest')

    monkeypatch.setattr(pipelines, 'DataHarmonization', MockHarmonization)
    monkeypatch.setattr(
        workflows.e2e_intake.batch_ingest,
        'execute_batch_ingest',
        mock_batch_ingest,
    )

    config = configs.RootConfig()
    workflows.execute_e2e_intake(config)

    assert calls == ['harmonize', 'batch_ingest']


# ----- `execute_e2e_experiment` tests
def test_execute_e2e_experiment_stage_sequence(monkeypatch):
    '''
    Given: A RootConfig instance.
    When: `execute_e2e_experiment` is executed.
    Then: Call world-grid, harmonize, ingest, prepare, and train.
    '''
    calls = []

    class MockWorldGrid:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self):
            calls.append('world_grid')

    class MockHarmonization:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self):
            calls.append('harmonize')

    class MockIngestion:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self):
            calls.append('ingest')

    class MockPreparation:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self):
            calls.append('prepare')

    class MockTraining:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self):
            calls.append('train')

    monkeypatch.setattr(pipelines, 'WorldGridGeneration', MockWorldGrid)
    monkeypatch.setattr(pipelines, 'DataHarmonization', MockHarmonization)
    monkeypatch.setattr(pipelines, 'DataIngestion', MockIngestion)
    monkeypatch.setattr(pipelines, 'DataPreparation', MockPreparation)
    monkeypatch.setattr(pipelines, 'ModelTraining', MockTraining)

    config = configs.RootConfig()
    workflows.execute_e2e_experiment(config)

    assert calls == ['world_grid', 'harmonize', 'ingest', 'prepare', 'train']
