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
import landseg.execution.preflight as preflight


# ----- `inspect_target` tests
def test_inspect_target_success(tmp_path, monkeypatch):
    '''
    Given: An execution target whose prerequisite and probes pass.
    When: `inspect_target()` is called.
    Then: Return a READY result with a passing prerequisite probe.
    '''
    config = configs.RootConfig()
    config.execution.exp_root = str(tmp_path / 'exp')
    monkeypatch.setattr(
        preflight.prerequisites,
        'check_target_prerequisites',
        lambda target, paths: [
            preflight.prerequisites.PrerequisiteCheck.passed(
                target=target,
                upstream_target='world-grid',
                message=f'Prerequisites verified for {target}',
            )
        ],
    )
    monkeypatch.setattr(
        preflight.probes,
        'raw_dataset',
        lambda cfg: preflight.schema.ProbeResult(
            pid='source_dataset_manifest',
            category='Dataset',
            status=preflight.ProbeStatus.PASS,
            message='Mock manifest valid',
        ),
    )

    result = preflight.inspect_target('data-harmonize', config)

    assert isinstance(result, preflight.PreflightResult)
    assert result.target == 'data-harmonize'
    assert result.status == 'READY'
    assert result.is_ready is True
    assert len(result.errors) == 0

    assert any(
        p.pid == 'pipeline_prerequisites'
        and p.status == preflight.ProbeStatus.PASS
        and 'data-harmonize' in p.message
        for p in result.probes
    )


def test_inspect_target_failure(tmp_path, monkeypatch):
    '''
    Given: An execution target whose prerequisite checks fail.
    When: `inspect_target()` is called.
    Then: Return a BLOCKED result capturing failure in probe errors.
    '''
    config = configs.RootConfig()
    config.execution.exp_root = str(tmp_path / 'exp')
    monkeypatch.setattr(
        preflight.prerequisites,
        'check_target_prerequisites',
        lambda target, paths: [
            preflight.prerequisites.PrerequisiteCheck.failed(
                target=target,
                upstream_target='world-grid',
                message='Missing upstream artifact',
            )
        ],
    )

    result = preflight.inspect_target('data-harmonize', config)

    assert isinstance(result, preflight.PreflightResult)
    assert result.target == 'data-harmonize'
    assert result.status == 'BLOCKED'
    assert result.is_ready is False
    assert len(result.errors) >= 1
    assert any('Missing upstream artifact' in err for err in result.errors)

    lineage_probe = next(
        p for p in result.probes if p.pid == 'pipeline_prerequisites'
    )
    assert lineage_probe.status == preflight.ProbeStatus.FAIL


# ----- probe unit tests
def test_probe_hardware():
    '''
    Given: A RootConfig with GPU check enabled.
    When: `hardware_info()` is called.
    Then: Return a list containing the cuda_device probe.
    '''
    cfg = configs.RootConfig()
    results = preflight.probes.hardware_info(cfg)
    assert len(results) >= 1
    assert results[0].category == 'Hardware'
    assert results[0].pid == 'cuda_device'
    assert 'torch_version' in results[0].details


def test_probe_filesystem(tmp_path):
    '''
    Given: A writable directory path and file target.
    When: `dir_writable()` and `target_file_exists()` are called.
    Then: Return passing probe results with category Filesystem.
    '''
    dir_result = preflight.probes.dir_writable(str(tmp_path), 'test_dir')
    assert dir_result.category == 'Filesystem'
    assert dir_result.pid == 'test_dir'
    assert dir_result.status == preflight.ProbeStatus.PASS

    file_result = preflight.probes.target_file_exists(
        str(tmp_path / 'out.json'), 'test_file'
    )
    assert file_result.category == 'Filesystem'
    assert file_result.pid == 'test_file'
    assert file_result.status == preflight.ProbeStatus.PASS


def test_probe_spatial():
    '''
    Given: A RootConfig with world grid parameters.
    When: Spatial probes are invoked.
    Then: Return valid probe results with category Spatial.
    '''
    cfg = configs.RootConfig()
    ref_result = preflight.probes.spatial_reference(cfg)
    crs_result = preflight.probes.crs_info(cfg)
    pix_result = preflight.probes.pixel_size(cfg)

    assert ref_result.category == 'Spatial'
    assert ref_result.pid == 'world_grid_reference'
    assert crs_result.pid == 'crs'
    assert pix_result.pid == 'pixel_size'


def test_probe_ledger(tmp_path):
    '''
    Given: A RootConfig pointing to a temporary experiment root.
    When: `ingestion_pool_state()` is called.
    Then: Return ledger diagnostic probe records.
    '''
    cfg = configs.RootConfig()
    cfg.execution.exp_root = str(tmp_path / 'exp')
    artifact_paths = artifacts.ArtifactPaths.from_config(cfg)
    result = preflight.probes.ingestion_pool_state(artifact_paths)
    assert result.category == 'Ledger'
    assert result.pid == 'ingested_blocks_pool'
    assert result.status == preflight.ProbeStatus.PASS


def test_probe_model():
    '''
    Given: A RootConfig with model architecture specifications.
    When: `model_body()` is called.
    Then: Verify model_body probe passes with category Model.
    '''
    cfg = configs.RootConfig()
    result = preflight.probes.model_body(cfg)
    assert result.category == 'Model'
    assert result.pid == 'model_body'
    assert result.status == preflight.ProbeStatus.PASS


def test_probe_policy():
    '''
    Given: A RootConfig with ingestion collision policy.
    When: `collision_policy()` is called.
    Then: Return collision_policy probe with category Policy.
    '''
    cfg = configs.RootConfig()
    result = preflight.probes.collision_policy(cfg)
    assert result.category == 'Policy'
    assert result.pid == 'collision_policy'


def test_inspect_target():
    '''
    Given: An execution target (batch-ingest).
    When: `inspect_target()` is invoked.
    Then: Return a valid PreflightResult with diagnostic probes.
    '''
    cfg = configs.RootConfig()
    result = preflight.inspect_target('batch-ingest', cfg)
    assert isinstance(result, preflight.PreflightResult)
    assert result.target == 'batch-ingest'
    probe_ids = [p.pid for p in result.probes]
    assert 'ingestion_output' in probe_ids
    assert 'collision_policy' in probe_ids
    assert 'past_harmonization_runs' in probe_ids
    assert 'past_ingestion_runs' in probe_ids
    assert 'pending_ingestion' in probe_ids
    assert 'ingested_blocks_pool' in probe_ids


def test_probe_canonical_ordering():
    '''
    Given: Pipeline targets with multiple diagnostic categories.
    When: Diagnostic probes are inspected.
    Then: Probes follow canonical category ordering.
    '''
    cfg = configs.RootConfig()
    category_priority = {
        'Lineage': 0,
        'Filesystem': 1,
        'Contract': 2,
        'Dataset': 2,
        'Policy': 2,
        'Model': 2,
        'Spatial': 2,
        'Ledger': 3,
        'Hardware': 4,
    }
    targets = (
        'world-grid',
        'data-harmonize',
        'batch-ingest',
        'model-train',
    )
    for target in targets:
        result = preflight.inspect_target(target, cfg)
        priorities = [
            category_priority.get(p.category, 99) for p in result.probes
        ]
        assert priorities == sorted(priorities)


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
                pid='probe_1',
                category='hardware',
                status=preflight.ProbeStatus.PASS,
                message='GPU available',
            ),
            preflight.ProbeResult(
                pid='probe_2',
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
                pid='pipeline_prerequisites',
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
                    pid='chk',
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
                pid='p1',
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
def test_run_preflight_dispatch_single_target(tmp_path):
    '''
    Given: A valid RootConfig with target='world-grid'.
    When: `run_preflight()` is invoked.
    Then: Return a PreflightResult for world-grid and export report.
    '''
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
    Then: Return a list of PreflightResults for all supported targets.
    '''
    monkeypatch.setattr(
        preflight.prerequisites,
        'check_target_prerequisites',
        lambda target, paths: [],
    )
    cfg = configs.RootConfig()
    cfg.execution.exp_root = str(tmp_path / 'exp')

    results = preflight.run_preflight(cfg, target='all')

    assert isinstance(results, list)
    assert len(results) == len(preflight.SUPPORTED_TARGETS)
    targets = [r.target for r in results]
    assert 'world-grid' in targets
    assert 'model-train' in targets
    assert 'batch-ingest' in targets
    assert 'diagnose-overfit' in targets


def test_run_preflight_strict_mode_failure(tmp_path, monkeypatch):
    '''
    Given: A failing pipeline and `strict=True`.
    When: `run_preflight()` is invoked.
    Then: Raise a RuntimeError due to strict mode enforcement.
    '''
    monkeypatch.setattr(
        preflight.prerequisites,
        'check_target_prerequisites',
        lambda target, paths: [
            preflight.prerequisites.PrerequisiteCheck.failed(
                target=target,
                upstream_target='world-grid',
                message='Missing source dataset',
            )
        ],
    )
    cfg = configs.RootConfig()
    cfg.execution.exp_root = str(tmp_path / 'exp')
    cfg.execution.preflight_strict_mode = True

    with pytest.raises(
        RuntimeError,
        match='Strict preflight validation failed'
    ):
        preflight.run_preflight(cfg, target='data-harmonize')


def test_run_preflight_dispatch_unknown_target():
    '''
    Given: An unsupported target string.
    When: `run_preflight()` is invoked.
    Then: Raise a KeyError with allowed targets.
    '''
    cfg = configs.RootConfig()
    with pytest.raises(KeyError, match='Target "unknown-target" not supported'):
        preflight.run_preflight(cfg, target='unknown-target')
