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

'''Unit tests for public `landseg.geopipe.stages` runners.'''

# standard imports
import pathlib
import types
import typing
# third-party imports
import pytest
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.geopipe as geopipe
import landseg.geopipe.grid as grid_mod
import landseg.geopipe.harmonize as harmonize_mod
import landseg.geopipe.ingest as ingest_mod
import landseg.geopipe.prepare as prepare_mod


# ----- world grid stage tests
def test_run_world_grid_stage_success_and_failure(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    '''
    Given: World grid config and output artifact paths.
    When: Running `geopipe.run_world_grid_stage` cleanly or with errors.
    Then: Persist grid report with `SUCCESS` or `FAILED` status
        accordingly.
    '''
    grid_paths = artifacts.WorldGridPaths(tmp_path / 'world_grids')
    grid_paths.init_pipeline_folders()
    wg_cfg = configs.RootConfig().data.world_grid

    class _FakeGrid(types.SimpleNamespace):
        '''Minimal grid stub with fixed tile count.'''

        def __len__(self) -> int:
            return 4

    grid_fp = tmp_path / 'world_grids' / 'grid.json'
    mock_grid = _FakeGrid(
        gid='gid_1',
        crs='EPSG:3161',
        pixel_size=(20.0, 20.0),
        tile_size=(256, 256),
        tile_stride=(128, 128),
    )
    monkeypatch.setattr(
        grid_mod,
        'prepare_world_grid',
        lambda _cfg: (False, grid_fp, mock_grid),
    )

    geopipe.run_world_grid_stage(wg_cfg, grid_paths)
    report = artifacts.Controller[dict](grid_paths.report).fetch()
    assert report and report['status'] == 'SUCCESS'
    assert report['total_tiles'] == 4

    def _boom(_cfg):
        raise RuntimeError('grid failure')

    monkeypatch.setattr(grid_mod, 'prepare_world_grid', _boom)
    with pytest.raises(RuntimeError, match='grid failure'):
        geopipe.run_world_grid_stage(wg_cfg, grid_paths)
    failed_report = artifacts.Controller[dict](grid_paths.report).fetch()
    assert failed_report and failed_report['status'] == 'FAILED'


# ----- data harmonization stage tests
def test_run_data_harmonization_stage_success_and_failure(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    '''
    Given: Harmonization config, artifact paths, and grid report path.
    When: Running `geopipe.run_data_harmonization_stage`.
    Then: Update `harmonization_runs.json` and persist `SUCCESS` or
        `FAILED` report.
    '''
    harm_paths = artifacts.HarmonizationPaths.allocate_run(tmp_path / 'harmonized')
    harm_paths.init_pipeline_folders()
    harm_cfg = configs.RootConfig().data.harmonization
    manifest_fp = tmp_path / 'manifest.json'
    manifest_fp.write_text('[]', encoding='utf-8')
    harm_cfg.dataset_manifest = str(manifest_fp)
    grid_report = tmp_path / 'world_grids' / 'report.json'

    monkeypatch.setattr(
        harmonize_mod,
        'build_harmonization_context',
        lambda *_a, **_kw: types.SimpleNamespace(collided_run_uid=None),
    )
    called: list[bool] = []
    monkeypatch.setattr(
        harmonize_mod,
        'run_data_harmonization',
        lambda *_a, **_kw: called.append(True),
    )

    geopipe.run_data_harmonization_stage(harm_cfg, harm_paths, grid_report)
    assert called == [True]
    report = artifacts.Controller[dict](harm_paths.report).fetch()
    assert report and report['status'] == 'SUCCESS'
    runs = artifacts.Controller[dict](harm_paths.runs_manifest).fetch()
    assert runs and any(r['status'] == 'SUCCESS' for r in runs.values())

    def _fail(*_a, **_kw):
        raise RuntimeError('harmonize crash')

    monkeypatch.setattr(harmonize_mod, 'run_data_harmonization', _fail)
    with pytest.raises(RuntimeError, match='harmonize crash'):
        geopipe.run_data_harmonization_stage(harm_cfg, harm_paths, grid_report)
    failed = artifacts.Controller[dict](harm_paths.report).fetch()
    assert failed and failed['status'] == 'FAILED'


# ----- data ingestion stage tests
def test_run_data_ingestion_stage_success_missing_and_failure(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    '''
    Given: Ingestion config and upstream/downstream artifact paths.
    When: Running `geopipe.run_data_ingestion_stage` with pending or
        failing batches.
    Then: Resolve batches, invoke stage runner, update ledger, and
        mark `FAILED` on errors.
    '''
    harm_paths = artifacts.HarmonizationPaths(tmp_path / 'harmonized')
    ing_paths = artifacts.IngestionPaths.allocate_run(tmp_path / 'ingested')
    ing_paths.init_pipeline_folders()
    ing_cfg = configs.RootConfig().data.ingestion

    monkeypatch.setattr(
        ingest_mod,
        'resolve_pending_ingestion_batches',
        lambda *_a, **_kw: [{'run_uid': 'u1', 'run_id': 'run_0001'}],
    )
    monkeypatch.setattr(
        ingest_mod,
        'build_ingestion_context',
        lambda *_a, **_kw: types.SimpleNamespace(
            harmonization_run_id='run_0001',
            run_uid='u1',
        ),
    )
    seen_policies: list[artifacts.LifecyclePolicy] = []
    monkeypatch.setattr(
        ingest_mod,
        'run_data_ingestion',
        lambda *_a, policy, **_kw: seen_policies.append(policy),
    )

    geopipe.run_data_ingestion_stage(
        ing_cfg,
        harm_paths,
        ing_paths,
        harmonization_run=ing_cfg.harmonization_run,
        rebuild=True,
    )
    assert seen_policies == [artifacts.LifecyclePolicy.REBUILD]
    report = artifacts.Controller[dict](ing_paths.report).fetch()
    assert report and report['status'] == 'SUCCESS'
    runs = artifacts.Controller[dict](ing_paths.runs_manifest).fetch()
    assert runs and any(r['status'] == 'SUCCESS' for r in runs.values())

    # no pending batches when record not supplied -> ArtifactError + FAILED
    monkeypatch.setattr(
        ingest_mod,
        'resolve_pending_ingestion_batches',
        lambda *_a, **_kw: [],
    )
    with pytest.raises(artifacts.ArtifactError, match='No pending harmonization'):
        geopipe.run_data_ingestion_stage(ing_cfg, harm_paths, ing_paths)
    failed = artifacts.Controller[dict](ing_paths.report).fetch()
    assert failed and failed['status'] == 'FAILED'


# ----- data preparation stage tests
def test_run_data_preparation_stage_success_and_failure(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    '''
    Given: Preparation config and ingestion/preparation artifact paths.
    When: Running `geopipe.run_data_preparation_stage`.
    Then: Build context, forward lifecycle policy, and persist report
        status.
    '''
    ing_paths = artifacts.IngestionPaths(tmp_path / 'ingested')
    prep_paths = artifacts.PreparationPaths(tmp_path / 'prepared')
    prep_paths.init_pipeline_folders()
    prep_cfg = configs.RootConfig().data.preparation
    prep_cfg.rebuild = False

    monkeypatch.setattr(
        prepare_mod,
        'build_preparation_context',
        lambda *_a, **_kw: object(),
    )
    seen: list[typing.Any] = []
    monkeypatch.setattr(
        prepare_mod,
        'run_data_preparation',
        lambda *_a, policy, **_kw: seen.append(policy),
    )

    geopipe.run_data_preparation_stage(prep_cfg, ing_paths, prep_paths)
    assert seen == [artifacts.LifecyclePolicy.BUILD_IF_MISSING]
    report = artifacts.Controller[dict](prep_paths.report).fetch()
    assert report and report['status'] == 'SUCCESS'

    def _boom(*_a, **_kw):
        raise RuntimeError('prep boom')

    monkeypatch.setattr(prepare_mod, 'run_data_preparation', _boom)
    with pytest.raises(RuntimeError, match='prep boom'):
        geopipe.run_data_preparation_stage(prep_cfg, ing_paths, prep_paths)
    failed = artifacts.Controller[dict](prep_paths.report).fetch()
    assert failed and failed['status'] == 'FAILED'
