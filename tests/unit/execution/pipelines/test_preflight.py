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

# standard imports
import os
# third-party imports
import pytest
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.pipelines as pipelines
import landseg.execution.pipelines.base as base
import landseg.execution.preflight as preflight


# ----- test helper classes
class _DummySuccessPipeline(base.BasePipeline):
    '''Concrete pipeline implementation with passing validation.'''

    pipeline_name = 'dummy-pipeline'

    def _create_logger(self):
        return None

    def _resolve_pipeline_paths(self):
        self.pipeline_paths = None

    def _build_context(self):
        return None

    def run(self):
        return 'success'

    def validate(self):
        pass


class _DummyFailingPipeline(base.BasePipeline):
    '''Concrete pipeline implementation that raises on validation.'''

    pipeline_name = 'failing-pipeline'

    def _create_logger(self):
        return None

    def _resolve_pipeline_paths(self):
        self.pipeline_paths = None

    def _build_context(self):
        return None

    def run(self):
        return 'failed'

    def validate(self):
        raise RuntimeError('Missing upstream artifact')


# ----- `inspect_target` tests
def test_inspect_target_success():
    '''
    Given: A pipeline whose `validate()` method succeeds without error.
    When: `inspect_target()` is called on the pipeline runner.
    Then: Return a READY result with a passing prerequisite probe.
    '''
    config = configs.RootConfig()
    runner = _DummySuccessPipeline(config)

    result = preflight.inspect_target(runner)

    assert isinstance(result, preflight.PreflightResult)
    assert result.target == 'dummy-pipeline'
    assert result.status == 'READY'
    assert result.is_ready is True
    assert len(result.errors) == 0

    assert any(
        p.probe_id == 'pipeline_prerequisites'
        and p.status == preflight.ProbeStatus.PASS
        and 'dummy-pipeline' in p.message
        for p in result.probes
    )


def test_inspect_target_failure():
    '''
    Given: A pipeline whose `validate()` method raises a RuntimeError.
    When: `inspect_target()` is called on the pipeline runner.
    Then: Return a BLOCKED result capturing the failure in probe errors.
    '''
    config = configs.RootConfig()
    runner = _DummyFailingPipeline(config)

    result = preflight.inspect_target(runner)

    assert isinstance(result, preflight.PreflightResult)
    assert result.target == 'failing-pipeline'
    assert result.status == 'BLOCKED'
    assert result.is_ready is False
    assert len(result.errors) == 1
    assert 'Missing upstream artifact' in result.errors[0]

    lineage_probe = next(
        p for p in result.probes if p.probe_id == 'pipeline_prerequisites'
    )
    assert lineage_probe.status == preflight.ProbeStatus.FAIL
    assert lineage_probe.details.get('error_type') == 'RuntimeError'


# ----- probe unit tests
def test_probe_hardware():
    '''
    Given: A call to `probe_hardware`.
    When: `check_gpu` is True.
    Then: Return a list containing the cuda_device probe.
    '''
    telemetry = {}
    results = preflight.probes.probe_hardware(
        check_gpu=True, telemetry=telemetry
    )
    assert len(results) >= 1
    assert results[0].category == 'hardware'
    assert results[0].probe_id == 'cuda_device'
    assert 'torch_version' in telemetry


def test_probe_storage():
    '''
    Given: A pipeline with pipeline_paths.
    When: `probe_storage` is called.
    Then: Return an output_directory probe result.
    '''
    cfg = configs.RootConfig()
    dp = pipelines.DataPreparation(cfg)
    results = preflight.probes.probe_storage(dp)
    assert len(results) == 1
    assert results[0].category == 'storage'
    assert results[0].probe_id == 'output_directory'


def test_probe_spatial():
    '''
    Given: A target and RootConfig.
    When: `probe_spatial` is called.
    Then: Return crs_validity, pixel_resolution, and block_dimensions.
    '''
    cfg = configs.RootConfig()
    results = preflight.probes.probe_spatial('world-grid', cfg)
    probe_ids = [p.probe_id for p in results]
    assert 'crs_validity' in probe_ids
    assert 'pixel_resolution' in probe_ids
    assert 'block_dimensions' in probe_ids


def test_probe_ledger(tmp_path):
    '''
    Given: A RootConfig pointing to a temporary experiment root.
    When: `probe_ledger` is called for data-ingest.
    Then: Return ledger diagnostic probe records.
    '''
    cfg = configs.RootConfig()
    cfg.execution.exp_root = str(tmp_path / 'exp')
    results = preflight.probes.probe_ledger('data-ingest', cfg)
    assert any(p.probe_id == 'harmonize_ledger' for p in results)


def test_probe_model():
    '''
    Given: A model-train target.
    When: `probe_model` is called.
    Then: Verify backbone_registry and channel_compatibility probes.
    '''
    cfg = configs.RootConfig()
    results = preflight.probes.probe_model('model-train', cfg)
    probe_ids = [p.probe_id for p in results]
    assert 'backbone_registry' in probe_ids
    assert 'channel_compatibility' in probe_ids


def test_probe_study():
    '''
    Given: A study-sweep target.
    When: `probe_study` is called.
    Then: Return study_schema probe result.
    '''
    cfg = configs.RootConfig()
    results = preflight.probes.probe_study('study-sweep', cfg)
    assert any(p.probe_id == 'study_schema' for p in results)


def test_inspect_target():
    '''
    Given: An execution target (batch-ingest).
    When: `inspect_target` is invoked.
    Then: Return a valid PreflightResult with diagnostic probes.
    '''
    cfg = configs.RootConfig()
    result = preflight.inspect_target('batch-ingest', cfg)
    assert isinstance(result, preflight.PreflightResult)
    assert result.target == 'batch-ingest'
    probe_ids = [p.probe_id for p in result.probes]
    assert 'harmonize_ledger' in probe_ids
    assert 'pending_batch_queue' in probe_ids
    assert 'collision_policy' in probe_ids


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


# ----- pipeline runner properties tests
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
    monkeypatch.setattr(
        pipelines.WorldGridGeneration, '_create_logger', lambda self: None
    )
    monkeypatch.setattr(
        pipelines.DataHarmonization, '_create_logger', lambda self: None
    )
    monkeypatch.setattr(
        pipelines.DataIngestion, '_create_logger', lambda self: None
    )
    monkeypatch.setattr(
        pipelines.DataPreparation, '_create_logger', lambda self: None
    )
    monkeypatch.setattr(
        pipelines.ModelTraining, '_create_logger', lambda self: None
    )
    monkeypatch.setattr(
        pipelines.ModelEvaluation, '_create_logger', lambda self: None
    )

    cfg = configs.RootConfig()
    wg = pipelines.WorldGridGeneration(cfg)
    dh = pipelines.DataHarmonization(cfg)
    di = pipelines.DataIngestion(cfg)
    dp = pipelines.DataPreparation(cfg)
    mt = pipelines.ModelTraining(cfg)
    me = pipelines.ModelEvaluation(cfg)

    assert wg.pipeline_paths == wg.artifact_paths.world_grid
    assert dh.pipeline_paths == dh.artifact_paths.data_harmonization
    assert di.pipeline_paths == di.artifact_paths.data_ingestion
    assert dp.pipeline_paths == dp.artifact_paths.data_preparation
    assert mt.pipeline_paths == mt.artifact_paths.session
    assert me.pipeline_paths == me.artifact_paths.session


# ----- reporting and formatting tests
def test_format_preflight_report_single():
    '''
    Given: A single target PreflightResult.
    When: `format_preflight_report()` is invoked.
    Then: Produce an 80-column ASCII terminal dashboard.
    '''
    result = preflight.PreflightResult(
        target='model-train',
        status='READY',
        probes=[
            preflight.ProbeResult(
                probe_id='pipeline_prerequisites',
                category='lineage',
                status=preflight.ProbeStatus.PASS,
                message='Prerequisites verified for model-train',
            ),
        ],
    )

    text = preflight.format_preflight_report(result)

    assert 'PRE-FLIGHT READINESS CHECK: model-train' in text
    assert 'lineage' in text
    assert 'pipeline_prerequisites' in text
    assert 'PASS' in text
    assert 'STATUS: READY (0 errors, 0 warnings)' in text


def test_format_preflight_report_multi():
    '''
    Given: A list of PreflightResult instances across multiple targets.
    When: `format_preflight_report()` is invoked.
    Then: Include individual reports and an aggregate SYSTEM STATUS summary.
    '''
    results = [
        preflight.PreflightResult(
            target='world-grid',
            status='READY',
            probes=[],
        ),
        preflight.PreflightResult(
            target='model-train',
            status='BLOCKED',
            probes=[
                preflight.ProbeResult(
                    probe_id='chk',
                    category='lineage',
                    status=preflight.ProbeStatus.FAIL,
                    message='Missing file',
                )
            ],
        ),
    ]

    text = preflight.format_preflight_report(results)

    assert 'PRE-FLIGHT READINESS CHECK: world-grid' in text
    assert 'PRE-FLIGHT READINESS CHECK: model-train' in text
    assert 'SYSTEM STATUS: 1 READY | 1 BLOCKED' in text


# ----- export tests
def test_export_preflight_report_timestamped_uid(tmp_path):
    '''
    Given: A preflight result and configured exp_root.
    When: `export_preflight_report()` is executed.
    Then: Persist report JSON with timestamped UID under exp_root/preflight.
    '''
    exp_root = str(tmp_path / 'exp')
    cfg = configs.RootConfig()
    cfg.execution.exp_root = exp_root

    result = preflight.PreflightResult(
        target='model-train',
        status='READY',
        probes=[
            preflight.ProbeResult(
                probe_id='p1',
                category='lineage',
                status=preflight.ProbeStatus.PASS,
                message='All ok',
            )
        ],
    )

    report_fp, uid = preflight.export_preflight_report(
        result, cfg, 'model-train'
    )

    assert os.path.exists(report_fp)
    assert f'preflight_report_{uid}.json' in report_fp
    assert os.path.dirname(report_fp) == os.path.join(exp_root, 'preflight')

    report_data = artifacts.Controller[dict].load_json_or_fail(
        report_fp
    ).fetch()
    assert report_data['uid'] == uid
    assert report_data['target'] == 'model-train'
    assert report_data['status'] == 'READY'
    assert 'timestamp' in report_data


# ----- `run_preflight` dispatch tests
def test_run_preflight_dispatch_single_target(tmp_path, monkeypatch):
    '''
    Given: A valid RootConfig with target='world-grid'.
    When: `run_preflight()` is invoked.
    Then: Return a PreflightResult for world-grid and export report.
    '''
    monkeypatch.setattr(
        pipelines.WorldGridGeneration, 'validate', lambda self: None
    )
    cfg = configs.RootConfig()
    cfg.execution.exp_root = str(tmp_path / 'exp')

    result = preflight.run_preflight(cfg, target='world-grid')

    assert isinstance(result, preflight.PreflightResult)
    assert result.target == 'world-grid'
    assert result.status == 'READY'


def test_run_preflight_dispatch_all_targets(tmp_path, monkeypatch):
    '''
    Given: A valid RootConfig with target='all'.
    When: `run_preflight()` is invoked.
    Then: Return a list of PreflightResults for all 10 targets.
    '''
    monkeypatch.setattr(
        pipelines.WorldGridGeneration, 'validate', lambda self: None
    )
    monkeypatch.setattr(
        pipelines.DataHarmonization, 'validate', lambda self: None
    )
    monkeypatch.setattr(
        pipelines.DataIngestion, 'validate', lambda self: None
    )
    monkeypatch.setattr(
        pipelines.DataPreparation, 'validate', lambda self: None
    )
    monkeypatch.setattr(
        pipelines.ModelTraining, 'validate', lambda self: None
    )
    monkeypatch.setattr(
        pipelines.ModelEvaluation, 'validate', lambda self: None
    )
    cfg = configs.RootConfig()
    cfg.execution.exp_root = str(tmp_path / 'exp')

    results = preflight.run_preflight(cfg, target='all')

    assert isinstance(results, list)
    assert len(results) == 10
    targets = [r.target for r in results]
    assert 'world-grid' in targets
    assert 'model-train' in targets
    assert 'batch-ingest' in targets
    assert 'study-sweep' in targets


def test_run_preflight_strict_mode_failure(tmp_path, monkeypatch):
    '''
    Given: A failing pipeline and `strict=True`.
    When: `run_preflight()` is invoked.
    Then: Raise a RuntimeError due to strict mode enforcement.
    '''
    def _fail_val(self):
        raise RuntimeError('Missing source dataset')

    monkeypatch.setattr(pipelines.WorldGridGeneration, 'validate', _fail_val)
    cfg = configs.RootConfig()
    cfg.execution.exp_root = str(tmp_path / 'exp')
    cfg.command.preflight.strict = True

    with pytest.raises(RuntimeError, match='Strict preflight validation failed'):
        preflight.run_preflight(cfg, target='world-grid')


def test_run_preflight_dispatch_unknown_target():
    '''
    Given: An unsupported target string.
    When: `run_preflight()` is invoked.
    Then: Raise a KeyError with allowed targets.
    '''
    cfg = configs.RootConfig()
    with pytest.raises(KeyError, match='Target "unknown-target" not supported'):
        preflight.run_preflight(cfg, target='unknown-target')
