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

'''Unit tests for standalone pre-flight execution validation module.'''

# local imports
import landseg.configs as configs
import landseg.execution.pipelines as pipelines
import landseg.execution.pipelines.base as base
import landseg.execution.preflight as preflight


# ----- test helper classes
class _DummySuccessPipeline(base.Pipeline):
    '''Concrete pipeline implementation with passing validation.'''

    pipeline_name = 'dummy-pipeline'

    def _create_logger(self):
        return None

    def run(self):
        return 'success'

    def validate(self):
        pass


class _DummyFailingPipeline(base.Pipeline):
    '''Concrete pipeline implementation that raises on validation.'''

    pipeline_name = 'failing-pipeline'

    def _create_logger(self):
        return None

    def run(self):
        return 'failed'

    def validate(self):
        raise RuntimeError('Missing upstream artifact')


# ----- `inspect_pipeline` tests
def test_inspect_pipeline_success():
    '''
    Given: A pipeline whose `validate()` method succeeds without error.
    When: `inspect_pipeline()` is called on the pipeline runner.
    Then: Return a READY result with a passing prerequisite probe.
    '''
    config = configs.RootConfig()
    runner = _DummySuccessPipeline(config)

    result = preflight.inspect_pipeline(runner)

    assert isinstance(result, preflight.PreflightResult)
    assert result.target == 'dummy-pipeline'
    assert result.status == 'READY'
    assert result.is_ready is True
    assert len(result.errors) == 0
    assert len(result.probes) == 1

    probe = result.probes[0]
    assert probe.probe_id == 'pipeline_prerequisites'
    assert probe.status == preflight.ProbeStatus.PASS
    assert 'dummy-pipeline' in probe.message


def test_inspect_pipeline_failure():
    '''
    Given: A pipeline whose `validate()` method raises a RuntimeError.
    When: `inspect_pipeline()` is called on the pipeline runner.
    Then: Return a BLOCKED result capturing the failure in probe errors.
    '''
    config = configs.RootConfig()
    runner = _DummyFailingPipeline(config)

    result = preflight.inspect_pipeline(runner)

    assert isinstance(result, preflight.PreflightResult)
    assert result.target == 'failing-pipeline'
    assert result.status == 'BLOCKED'
    assert result.is_ready is False
    assert len(result.errors) == 1
    assert 'Missing upstream artifact' in result.errors[0]
    assert result.probes[0].status == preflight.ProbeStatus.FAIL
    assert result.probes[0].details.get('error_type') == 'RuntimeError'


def test_preflight_result_serialization():
    '''
    Given: A PreflightResult instance with PASS and WARN probes.
    When: `as_dict()` is invoked.
    Then: Return a valid JSON-serializable dictionary representation.
    '''
    result = preflight.PreflightResult(
        target='test-pipeline',
        status='READY',
        probes=[
            preflight.ProbeResult(
                probe_id='probe_1',
                category='hardware',
                status=preflight.ProbeStatus.PASS,
                message='GPU available',
            ),
            preflight.ProbeResult(
                probe_id='probe_2',
                category='lineage',
                status=preflight.ProbeStatus.WARN,
                message='Pending batches detected',
            ),
        ],
        telemetry={'batch_size': 32},
    )

    data = result.as_dict()

    assert data['target'] == 'test-pipeline'
    assert data['status'] == 'READY'
    assert data['is_ready'] is True
    assert len(data['probes']) == 2
    assert data['probes'][0]['status'] == 'PASS'
    assert data['probes'][1]['status'] == 'WARN'
    assert data['warnings'] == ['Pending batches detected']
    assert data['telemetry']['batch_size'] == 32


def test_pipeline_runner_class_names():
    '''
    Given: The 6 canonical atomic pipeline runner classes.
    When: Inspecting their `pipeline_name` ClassVar attributes.
    Then: Match the expected canonical command identifiers.
    '''
    expected = {
        pipelines.WorldGridGeneration: 'world-grid',
        pipelines.DataHarmonization: 'data-harmonize',
        pipelines.DataIngestion: 'data-ingest',
        pipelines.DataPreparation: 'data-prepare',
        pipelines.ModelTraining: 'model-train',
        pipelines.ModelEvaluation: 'model-evaluate',
    }

    for runner_cls, expected_name in expected.items():
        assert runner_cls.pipeline_name == expected_name


def test_pipeline_default_path_resolution(monkeypatch):
    '''
    Given: Instantiated pipeline runners with a valid RootConfig.
    When: Accessing `pipeline_paths`.
    Then: Resolve automatically to expected domain paths from ArtifactPaths.
    '''
    monkeypatch.setattr(pipelines.WorldGridGeneration, '_create_logger', lambda self: None)
    monkeypatch.setattr(pipelines.DataHarmonization, '_create_logger', lambda self: None)
    monkeypatch.setattr(pipelines.DataIngestion, '_create_logger', lambda self: None)
    monkeypatch.setattr(pipelines.DataPreparation, '_create_logger', lambda self: None)
    monkeypatch.setattr(pipelines.ModelTraining, '_create_logger', lambda self: None)
    monkeypatch.setattr(pipelines.ModelEvaluation, '_create_logger', lambda self: None)

    cfg = configs.RootConfig()
    wg = pipelines.WorldGridGeneration(cfg)
    dh = pipelines.DataHarmonization(cfg)
    di = pipelines.DataIngestion(cfg)
    dp = pipelines.DataPreparation(cfg)
    mt = pipelines.ModelTraining(cfg)
    me = pipelines.ModelEvaluation(cfg)

    assert wg.pipeline_paths is None
    assert dh.pipeline_paths == dh.artifact_paths.data_harmonization
    assert di.pipeline_paths == di.artifact_paths.data_ingestion
    assert dp.pipeline_paths == dp.artifact_paths.data_preparation
    assert mt.pipeline_paths == mt.artifact_paths.session
    assert me.pipeline_paths == me.artifact_paths.session


def test_run_preflight_dispatch_single_target(monkeypatch):
    '''
    Given: A valid RootConfig with target='world-grid'.
    When: `run_preflight()` is invoked.
    Then: Return a PreflightResult for world-grid.
    '''
    monkeypatch.setattr(pipelines.WorldGridGeneration, 'validate', lambda self: None)
    cfg = configs.RootConfig()

    result = preflight.run_preflight(cfg, target='world-grid')

    assert isinstance(result, preflight.PreflightResult)
    assert result.target == 'world-grid'
    assert result.status == 'READY'


def test_run_preflight_dispatch_all_targets(monkeypatch):
    '''
    Given: A valid RootConfig with target='all'.
    When: `run_preflight()` is invoked.
    Then: Return a list of PreflightResults for all 6 pipelines.
    '''
    monkeypatch.setattr(pipelines.WorldGridGeneration, 'validate', lambda self: None)
    monkeypatch.setattr(pipelines.DataHarmonization, 'validate', lambda self: None)
    monkeypatch.setattr(pipelines.DataIngestion, 'validate', lambda self: None)
    monkeypatch.setattr(pipelines.DataPreparation, 'validate', lambda self: None)
    monkeypatch.setattr(pipelines.ModelTraining, 'validate', lambda self: None)
    monkeypatch.setattr(pipelines.ModelEvaluation, 'validate', lambda self: None)
    cfg = configs.RootConfig()

    results = preflight.run_preflight(cfg, target='all')

    assert isinstance(results, list)
    assert len(results) == 6
    targets = [r.target for r in results]
    assert 'world-grid' in targets
    assert 'model-train' in targets


def test_run_preflight_dispatch_unknown_target():
    '''
    Given: An unsupported target string.
    When: `run_preflight()` is invoked.
    Then: Raise a KeyError with allowed targets.
    '''
    cfg = configs.RootConfig()
    try:
        preflight.run_preflight(cfg, target='unknown-target')
        assert False, 'Should have raised KeyError'
    except KeyError as err:
        assert 'Target "unknown-target" not supported' in str(err)

