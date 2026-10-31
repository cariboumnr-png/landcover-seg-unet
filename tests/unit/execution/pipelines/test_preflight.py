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
import types
# third-party imports
import pytest
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.pipelines as pipelines
import landseg.execution.preflight as preflight
import landseg.execution.preflight.catalog as catalog
import landseg.execution.preflight.checks as checks
import landseg.execution.preflight.context as context
import landseg.execution.preflight.reporter as reporter
import landseg.execution.preflight.schema as schema
import landseg.utils.validator as val


# ----- `inspect_target` tests
def test_inspect_target_success(tmp_path, monkeypatch):
    '''
    Given: An execution target whose registered probes all pass.
    When: `inspect_target()` is called.
    Then: Return a READY result with no errors.
    '''
    config = configs.RootConfig()
    config.execution.exp_root = str(tmp_path / 'exp')
    ctx = context.InspectionContext.from_config(config)

    passing_spec = catalog.ProbeSpec(
        pid='mock_probe',
        run=lambda c: schema.ProbeResult(
            pid='mock_probe',
            category=schema.ProbeCategory.LINEAGE,
            status=schema.ProbeStatus.PASS,
            message='Prerequisites verified for data-harmonize',
        ),
    )
    monkeypatch.setattr(
        catalog,
        'suite_for',
        lambda target: (passing_spec,),
    )

    result = preflight.inspect_target('data-harmonize', ctx)

    assert isinstance(result, schema.PreflightResult)
    assert result.target == 'data-harmonize'
    assert result.status == 'READY'
    assert result.is_ready is True
    assert len(result.errors) == 0
    assert result.probes[0].pid == 'mock_probe'


def test_inspect_target_failure(tmp_path, monkeypatch):
    '''
    Given: An execution target with a failing probe.
    When: `inspect_target()` is called.
    Then: Return a BLOCKED result capturing failure in probe errors.
    '''
    config = configs.RootConfig()
    config.execution.exp_root = str(tmp_path / 'exp')
    ctx = context.InspectionContext.from_config(config)

    failing_spec = catalog.ProbeSpec(
        pid='mock_failing_probe',
        run=lambda c: schema.ProbeResult(
            pid='mock_failing_probe',
            category=schema.ProbeCategory.LINEAGE,
            status=schema.ProbeStatus.FAIL,
            message='Missing upstream artifact',
        ),
    )
    monkeypatch.setattr(
        catalog,
        'suite_for',
        lambda target: (failing_spec,),
    )

    result = preflight.inspect_target('data-harmonize', ctx)

    assert isinstance(result, schema.PreflightResult)
    assert result.target == 'data-harmonize'
    assert result.status == 'BLOCKED'
    assert result.is_ready is False
    assert len(result.errors) == 1
    assert 'Missing upstream artifact' in result.errors[0]
    assert result.probes[0].status == schema.ProbeStatus.FAIL


# ----- probe unit tests
def test_probe_hardware():
    '''
    Given: A RootConfig with GPU check enabled.
    When: `hardware_info()` is called.
    Then: Return a list containing the cuda_device probe.
    '''
    cfg = configs.RootConfig()
    ctx = context.InspectionContext.from_config(cfg)
    results = checks.hardware.hardware_info(ctx)
    assert len(results) >= 1
    assert results[0].category == schema.ProbeCategory.HARDWARE
    assert results[0].pid == 'cuda_device'
    assert 'torch_version' in results[0].details


def test_probe_filesystem(tmp_path):
    '''
    Given: A writable directory path and a missing path.
    When: `dir_writable()` is called.
    Then: Return PASS for writable path and FAIL for undefined path.
    '''
    dir_result = checks.filesystem.dir_writable(tmp_path, 'test_dir')
    assert dir_result.category == schema.ProbeCategory.FILESYSTEM
    assert dir_result.pid == 'test_dir'
    assert dir_result.status == schema.ProbeStatus.PASS

    none_result = checks.filesystem.dir_writable(None, 'missing_dir')
    assert none_result.category == schema.ProbeCategory.FILESYSTEM
    assert none_result.pid == 'missing_dir'
    assert none_result.status == schema.ProbeStatus.FAIL


def test_probe_ledger(tmp_path):
    '''
    Given: A RootConfig pointing to a temporary experiment root.
    When: `ingested_pool_state()` is called.
    Then: Return ledger diagnostic probe records.
    '''
    cfg = configs.RootConfig()
    cfg.execution.exp_root = str(tmp_path / 'exp')
    ctx = context.InspectionContext.from_config(cfg)
    result = checks.ledger.ingested_pool_state(ctx)
    assert result.category == schema.ProbeCategory.LEDGER
    assert result.pid == 'ingested_blocks_pool'
    assert result.status == schema.ProbeStatus.PASS


def test_probe_model():
    '''
    Given: A RootConfig with model architecture specifications.
    When: `model_body()` is called.
    Then: Verify model_body probe passes with category Model.
    '''
    cfg = configs.RootConfig()
    ctx = context.InspectionContext.from_config(cfg)
    result = checks.model.model_body(ctx)
    assert result.category == schema.ProbeCategory.MODEL
    assert result.pid == 'model_body'
    assert result.status == schema.ProbeStatus.PASS


def test_probe_policy():
    '''
    Given: A RootConfig with ingestion collision policy.
    When: `collision_policy()` is called.
    Then: Return collision_policy probe with category Policy.
    '''
    cfg = configs.RootConfig()
    ctx = context.InspectionContext.from_config(cfg)
    result = checks.policy.collision_policy(ctx)
    assert result.category == schema.ProbeCategory.POLICY
    assert result.pid == 'collision_policy'


def test_probe_manifest_contract_unconfigured():
    '''
    Given: A RootConfig with empty dataset manifest path.
    When: `manifest_contract()` is called.
    Then: Return a FAIL probe result with category Contract.
    '''
    cfg = configs.RootConfig()
    cfg.data.harmonization.dataset_manifest = ''
    ctx = context.InspectionContext.from_config(cfg)
    result = checks.contracts.manifest_contract(ctx)
    assert result.category == schema.ProbeCategory.CONTRACT
    assert result.pid == 'manifest_contract'
    assert result.status == schema.ProbeStatus.FAIL
    assert 'not configured' in result.message


def test_probe_manifest_contract_success(monkeypatch):
    '''
    Given: A RootConfig with valid dataset manifest contract.
    When: `manifest_contract()` is called.
    Then: Return a PASS probe result with category Contract.
    '''
    cfg = configs.RootConfig()
    cfg.data.harmonization.dataset_manifest = 'dummy/manifest.json'
    monkeypatch.setattr(
        'landseg.geopipe.harmonize.validators.manifest_contract',
        lambda fpath: ({'name': 's2'},),
    )
    ctx = context.InspectionContext.from_config(cfg)
    result = checks.contracts.manifest_contract(ctx)
    assert result.category == schema.ProbeCategory.CONTRACT
    assert result.pid == 'manifest_contract'
    assert result.status == schema.ProbeStatus.PASS
    assert '1 valid entries' in result.message


def test_probe_manifest_contract_validation_error(monkeypatch):
    '''
    Given: A manifest raising ValidationError during validation.
    When: `manifest_contract()` is called.
    Then: Return a FAIL probe result capturing ValidationError.
    '''
    cfg = configs.RootConfig()
    cfg.data.harmonization.dataset_manifest = 'dummy/manifest.json'

    def _failing_validate(fpath):
        raise val.ValidationError('Duplicate output name')

    monkeypatch.setattr(
        'landseg.geopipe.harmonize.validators.manifest_contract',
        _failing_validate,
    )
    ctx = context.InspectionContext.from_config(cfg)
    result = checks.contracts.manifest_contract(ctx)
    assert result.category == schema.ProbeCategory.CONTRACT
    assert result.status == schema.ProbeStatus.FAIL
    assert 'Duplicate output name' in result.message


def test_probe_grid_reference_missing(tmp_path):
    '''
    Given: A non-existent world grid report path.
    When: `grid_artifact()` is called.
    Then: Return a FAIL probe result with category Contract.
    '''
    cfg = configs.RootConfig()
    cfg.execution.exp_root = str(tmp_path / 'exp')
    ctx = context.InspectionContext.from_config(cfg)
    result = checks.contracts.grid_artifact(ctx)
    assert result.category == schema.ProbeCategory.CONTRACT
    assert result.pid == 'grid_reference'
    assert result.status == schema.ProbeStatus.FAIL
    assert 'validation failed' in result.message


def test_probe_grid_reference_success(monkeypatch, tmp_path):
    '''
    Given: A valid world grid reference handler response.
    When: `grid_artifact()` is called.
    Then: Return a PASS probe result with category Contract.
    '''
    mock_grid = {'tile_1': 'window_1', 'tile_2': 'window_2'}
    monkeypatch.setattr(
        'landseg.geopipe.harmonize.validators.grid_reference',
        lambda report_fpath: (mock_grid, 'grid_utm17', 'grid.json'),
    )
    cfg = configs.RootConfig()
    cfg.execution.exp_root = str(tmp_path / 'exp')
    ctx = context.InspectionContext.from_config(cfg)
    result = checks.contracts.grid_artifact(ctx)
    assert result.category == schema.ProbeCategory.CONTRACT
    assert result.pid == 'grid_reference'
    assert result.status == schema.ProbeStatus.PASS
    assert 'grid_id: grid_utm17' in result.message
    assert 'tiles: 2' in result.message


def test_probe_grid_reference_failure(monkeypatch, tmp_path):
    '''
    Given: A handler raising an exception during grid validation.
    When: `grid_artifact()` is called.
    Then: Return a FAIL probe result capturing the error message.
    '''
    def _failing_validate(report_fpath):
        raise RuntimeError('Corrupt grid layout payload')

    monkeypatch.setattr(
        'landseg.geopipe.harmonize.validators.grid_reference',
        _failing_validate,
    )
    cfg = configs.RootConfig()
    cfg.execution.exp_root = str(tmp_path / 'exp')
    ctx = context.InspectionContext.from_config(cfg)
    result = checks.contracts.grid_artifact(ctx)
    assert result.category == schema.ProbeCategory.CONTRACT
    assert result.status == schema.ProbeStatus.FAIL
    assert 'Corrupt grid layout payload' in result.message


def test_probe_grid_inputs_success(monkeypatch):
    '''
    Given: A RootConfig with valid world grid inputs.
    When: `grid_specs()` is called.
    Then: Return a PASS probe result with category Contract.
    '''
    cfg = configs.RootConfig()
    cfg.data.world_grid.mode = 'manual'
    monkeypatch.setattr(
        'landseg.geopipe.grid.validators.grid_specs',
        lambda wg_cfg: types.SimpleNamespace(grid_id='grid_utm17', tiles=4),
    )
    ctx = context.InspectionContext.from_config(cfg)
    result = checks.contracts.grid_specs(ctx)
    assert result.category == schema.ProbeCategory.CONTRACT
    assert result.pid == 'grid_inputs'
    assert result.status == schema.ProbeStatus.PASS


def test_probe_grid_inputs_failure(monkeypatch):
    '''
    Given: A grid handler raising an exception during grid construction.
    When: `grid_specs()` is called.
    Then: Return a FAIL probe result capturing the error message.
    '''
    cfg = configs.RootConfig()

    def _failing_validate(wg_cfg):
        raise ValueError('Invalid reference raster: dummy.tif')

    monkeypatch.setattr(
        'landseg.geopipe.grid.validators.grid_specs',
        _failing_validate,
    )
    ctx = context.InspectionContext.from_config(cfg)
    result = checks.contracts.grid_specs(ctx)
    assert result.category == schema.ProbeCategory.CONTRACT
    assert result.pid == 'grid_inputs'
    assert result.status == schema.ProbeStatus.FAIL
    assert 'Invalid reference raster' in result.message


def test_probe_pool_grid_compatibility_success(monkeypatch, tmp_path):
    '''
    Given: A grid layout and compatible schema.
    When: `pool_compatibility()` is called.
    Then: Return a PASS probe result with category Contract.
    '''
    cfg = configs.RootConfig()
    paths = artifacts.ArtifactPaths(root=tmp_path / 'exp')
    ctx = context.InspectionContext.from_config(cfg, artifact_paths=paths)
    ctx.paths.world_grid.report.parent.mkdir(parents=True, exist_ok=True)
    ctx.paths.world_grid.report.touch()

    mock_grid = types.SimpleNamespace(gid='grid_utm17')
    monkeypatch.setattr(
        'landseg.geopipe.harmonize.validators.grid_reference',
        lambda fp: (mock_grid, 'grid_utm17', fp),
    )
    monkeypatch.setattr(
        'landseg.geopipe.ingest.validators.pool_compatibility',
        lambda fp, grid: None,
    )
    result = checks.contracts.pool_compatibility(ctx)
    assert result.category == schema.ProbeCategory.CONTRACT
    assert result.pid == 'pool_grid_compatibility'
    assert result.status == schema.ProbeStatus.PASS
    assert 'Pool schema not yet initialized' in result.message


def test_probe_pool_grid_compatibility_failure(monkeypatch, tmp_path):
    '''
    Given: A domain handler raising ArtifactError on grid mismatch.
    When: `pool_compatibility()` is called.
    Then: Return a FAIL probe result capturing the error message.
    '''
    cfg = configs.RootConfig()
    paths = artifacts.ArtifactPaths(root=tmp_path / 'exp')
    ctx = context.InspectionContext.from_config(cfg, artifact_paths=paths)
    ctx.paths.world_grid.report.parent.mkdir(parents=True, exist_ok=True)
    ctx.paths.world_grid.report.touch()

    mock_grid = types.SimpleNamespace(gid='grid_utm17')
    monkeypatch.setattr(
        'landseg.geopipe.harmonize.validators.grid_reference',
        lambda fp: (mock_grid, 'grid_utm17', fp),
    )

    def _failing_validate(fp, grid):
        raise artifacts.ArtifactError('Grid compatibility mismatch')

    monkeypatch.setattr(
        'landseg.geopipe.ingest.validators.pool_compatibility',
        _failing_validate,
    )
    result = checks.contracts.pool_compatibility(ctx)
    assert result.category == schema.ProbeCategory.CONTRACT
    assert result.pid == 'pool_grid_compatibility'
    assert result.status == schema.ProbeStatus.FAIL
    assert 'Grid compatibility mismatch' in result.message


def test_probe_preparation_contract_schema_missing(tmp_path):
    '''
    Given: A missing upstream ingestion schema path.
    When: `prepare_semantics()` is called.
    Then: Return a FAIL probe result with schema missing message.
    '''
    cfg = configs.RootConfig()
    paths = artifacts.ArtifactPaths(root=tmp_path / 'exp')
    ctx = context.InspectionContext.from_config(cfg, artifact_paths=paths)
    result = checks.contracts.prepare_semantics(ctx)
    assert result.category == schema.ProbeCategory.CONTRACT
    assert result.pid == 'preparation_contract'
    assert result.status == schema.ProbeStatus.FAIL
    assert 'Upstream ingestion schema not found' in result.message


def test_probe_preparation_contract_success(monkeypatch, tmp_path):
    '''
    Given: Valid schema and matching dataset preparation configuration.
    When: `prepare_semantics()` is called.
    Then: Return a PASS probe result with category Contract.
    '''
    cfg = configs.RootConfig()
    paths = artifacts.ArtifactPaths(root=tmp_path / 'exp')
    ctx = context.InspectionContext.from_config(cfg, artifact_paths=paths)
    ctx.paths.data_ingestion.data_blocks.schema.parent.mkdir(
        parents=True, exist_ok=True
    )
    ctx.paths.data_ingestion.data_blocks.schema.write_text(
        '{"schema_id": "schema"}', encoding='utf-8'
    )

    monkeypatch.setattr(
        artifacts.Controller,
        'fetch',
        lambda self: {'schema_id': 'schema'},
    )
    mock_features = types.SimpleNamespace(names=('b1', 'b2'))
    mock_targets = types.SimpleNamespace(head_names=('h1',))
    monkeypatch.setattr(
        'landseg.geopipe.prepare.validators.feature_channels',
        lambda schm, cfg_f: mock_features,
    )
    monkeypatch.setattr(
        'landseg.geopipe.prepare.validators.target_heads',
        lambda schm, cfg_t: mock_targets,
    )
    monkeypatch.setattr(
        'landseg.geopipe.prepare.validators.focal_head',
        lambda targets, focal: 'h1',
    )
    result = checks.contracts.prepare_semantics(ctx)
    assert result.category == schema.ProbeCategory.CONTRACT
    assert result.pid == 'preparation_contract'
    assert result.status == schema.ProbeStatus.PASS
    assert 'Preparation contract verified' in result.message


def test_probe_preparation_contract_failure(monkeypatch, tmp_path):
    '''
    Given: A domain handler raising KeyError on feature mismatch.
    When: `prepare_semantics()` is called.
    Then: Return a FAIL probe result capturing the error message.
    '''
    cfg = configs.RootConfig()
    paths = artifacts.ArtifactPaths(root=tmp_path / 'exp')
    ctx = context.InspectionContext.from_config(cfg, artifact_paths=paths)
    ctx.paths.data_ingestion.data_blocks.schema.parent.mkdir(
        parents=True, exist_ok=True
    )
    ctx.paths.data_ingestion.data_blocks.schema.write_text(
        '{"schema_id": "schema"}', encoding='utf-8'
    )

    monkeypatch.setattr(
        artifacts.Controller,
        'fetch',
        lambda self: {'schema_id': 'schema'},
    )

    def _failing_features(schm, cfg_f):
        raise KeyError('Requested feature band not found')

    monkeypatch.setattr(
        'landseg.geopipe.prepare.validators.feature_channels',
        _failing_features,
    )
    result = checks.contracts.prepare_semantics(ctx)
    assert result.category == schema.ProbeCategory.CONTRACT
    assert result.pid == 'preparation_contract'
    assert result.status == schema.ProbeStatus.FAIL
    assert 'Requested feature band not found' in result.message


def test_inspect_target():
    '''
    Given: An execution target (data-ingest).
    When: `inspect_target()` is invoked.
    Then: Return a valid PreflightResult with diagnostic probes.
    '''
    cfg = configs.RootConfig()
    ctx = context.InspectionContext.from_config(cfg)
    result = preflight.inspect_target('data-ingest', ctx)
    assert isinstance(result, schema.PreflightResult)
    assert result.target == 'data-ingest'
    probe_ids = [p.pid for p in result.probes]
    assert 'ingestion_output' in probe_ids
    assert 'collision_policy' in probe_ids
    assert 'pool_grid_compatibility' in probe_ids
    assert 'past_harmonization_runs' in probe_ids
    assert 'past_ingestion_runs' in probe_ids
    assert 'pending_ingestion' in probe_ids
    assert 'ingested_blocks_pool' in probe_ids


def test_probe_canonical_ordering():
    '''
    Given: Pipeline targets with registered diagnostic suites.
    When: Diagnostic probes are inspected.
    Then: Probes preserve the exact ordering declared in
        `catalog.SUITES`.
    '''
    cfg = configs.RootConfig()
    ctx = context.InspectionContext.from_config(cfg)
    targets = (
        'world-grid', 'data-harmonize', 'data-ingest', 'data-prepare',
        'model-train',
    )
    pid_map = {'hardware': 'cuda_device'}
    for target in targets:
        result = preflight.inspect_target(target, ctx)
        specs = catalog.suite_for(target)
        expected_pids = [pid_map.get(s.pid, s.pid) for s in specs]
        actual_pids = [p.pid for p in result.probes]
        assert actual_pids == expected_pids


def test_preflight_result_serialization():
    '''
    Given: A PreflightResult instance with PASS and WARN probes.
    When: `as_dict()` is invoked.
    Then: Return a valid JSON-serializable dictionary representation.
    '''
    result = schema.PreflightResult(
        target='test-pipeline',
        status='READY',
        probes=[
            schema.ProbeResult(
                pid='probe_1',
                category=schema.ProbeCategory.HARDWARE,
                status=schema.ProbeStatus.PASS,
                message='GPU available',
            ),
            schema.ProbeResult(
                pid='probe_2',
                category=schema.ProbeCategory.LINEAGE,
                status=schema.ProbeStatus.WARN,
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
    result = schema.PreflightResult(
        target='model-train',
        status='READY',
        probes=[
            schema.ProbeResult(
                pid='pipeline_prerequisites',
                category=schema.ProbeCategory.LINEAGE,
                status=schema.ProbeStatus.PASS,
                message='Prerequisites verified for model-train',
            ),
        ],
    )

    text = reporter.format_preflight_report(result)

    assert 'PRE-FLIGHT READINESS CHECK: model-train' in text
    assert 'Lineage' in text
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
        schema.PreflightResult(
            target='world-grid',
            status='READY',
            probes=[],
        ),
        schema.PreflightResult(
            target='model-train',
            status='BLOCKED',
            probes=[
                schema.ProbeResult(
                    pid='chk',
                    category=schema.ProbeCategory.LINEAGE,
                    status=schema.ProbeStatus.FAIL,
                    message='Missing file',
                )
            ],
        ),
    ]

    text = reporter.format_preflight_report(results)

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
    exp_root = tmp_path / 'exp'
    cfg = configs.RootConfig()
    cfg.execution.exp_root = str(exp_root)

    result = schema.PreflightResult(
        target='model-train',
        status='READY',
        probes=[
            schema.ProbeResult(
                pid='p1',
                category=schema.ProbeCategory.LINEAGE,
                status=schema.ProbeStatus.PASS,
                message='All ok',
            )
        ],
    )

    report_fp, uid = reporter.export_preflight_report(
        result, cfg, 'model-train'
    )

    assert report_fp.exists()
    assert report_fp.name == f'preflight_report_{uid}.json'
    assert report_fp.parent == exp_root / 'preflight'

    report_data = artifacts.Controller[dict](report_fp).fetch()
    assert report_data['uid'] == uid
    assert report_data['target'] == 'model-train'
    assert report_data['status'] == 'READY'
    assert 'timestamp' in report_data


# ----- `run_preflight` dispatch tests
def test_run_preflight_dispatch_single_target(tmp_path, dummy_data_paths):
    '''
    Given: A valid RootConfig with target='world-grid'.
    When: `run_preflight()` is invoked.
    Then: Return a PreflightResult for world-grid and export report.
    '''
    cfg = configs.RootConfig()
    cfg.execution.exp_root = str(tmp_path / 'exp')
    cfg.data.world_grid.ref_fpath = str(dummy_data_paths.extent)

    result = preflight.run_preflight(cfg, target='world-grid')

    assert isinstance(result, schema.PreflightResult)
    assert result.target == 'world-grid'
    assert result.status == 'READY'


def test_run_preflight_dispatch_all_targets(tmp_path):
    '''
    Given: A valid RootConfig with target='all'.
    When: `run_preflight()` is invoked.
    Then: Return a list of PreflightResults for all supported targets.
    '''
    cfg = configs.RootConfig()
    cfg.execution.exp_root = str(tmp_path / 'exp')

    results = preflight.run_preflight(cfg, target='all')

    assert isinstance(results, list)
    assert len(results) == len(catalog.SUPPORTED_TARGETS)
    targets = [r.target for r in results]
    assert 'world-grid' in targets
    assert 'model-train' in targets
    assert 'data-ingest' in targets
    assert 'diagnose-overfit' in targets


def test_run_preflight_strict_mode_failure(tmp_path):
    '''
    Given: A failing pipeline and `strict=True`.
    When: `run_preflight()` is invoked.
    Then: Raise a RuntimeError due to strict mode enforcement.
    '''
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
