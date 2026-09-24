# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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

'''Unit tests for ingestion context resolution.'''

# standard imports
import dataclasses
import json
import os
# third-party imports
import pytest
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.contracts as contracts
import landseg.geopipe.core as geo_core
import landseg.geopipe.ingest.context as ingest_context


# ----- private helpers
def _create_dummy_grid(fpath: str) -> geo_core.GridLayout:
    '''Create and persist a minimal grid layout artifact.'''
    spec = geo_core.GridSpec(
        crs='EPSG:32617',
        origin=(0.0, 1000.0),
        pixel_size=(10.0, 10.0),
        tile_size=(16, 16),
        tile_stride=(8, 8),
        grid_extent=(160.0, 160.0),
    )
    layout = geo_core.GridLayout(spec)
    ctrl = artifacts.PayloadController[
        list[list[int]], geo_core.GridMeta
    ](
        fpath,
        schema_id=geo_core.GridLayout.SCHEMA_ID,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING
    )
    ctrl.save(layout.to_payload())
    return layout


# ----- `IngestionContext` tests
def test_ingestion_context_properties(tmp_path):
    '''
    Given: Instantiated IngestionContext.
    When: Checking has_data and attempting attribute mutation.
    Then: has_data evaluates correctly and mutation raises error.
    '''
    grid_fp = str(tmp_path / 'grid.json')
    layout = _create_dummy_grid(grid_fp)
    ctx = ingest_context.IngestionContext(
        grid=layout,
        grid_fpath=grid_fp,
        domains={'landcover': '/data/lc.tif'},
        features='/data/features.vrt',
        labels='/data/labels.vrt',
        valid_mask_raster='/data/mask.tif',
    )
    assert ctx.has_data is True
    assert ctx.grid_fpath == grid_fp
    assert ctx.grid.gid == layout.gid
    with pytest.raises(dataclasses.FrozenInstanceError):
        # pylint: disable=dataclass-cannot-be-modified
        ctx.grid_fpath = 'other_path'


def test_build_ingestion_context_success(tmp_path):
    '''
    Given: World grid, runs manifest, and harmonization report.
    When: `build_ingestion_context` is executed.
    Then: Successfully resolve IngestionContext with loaded GridLayout.
    '''
    grid_fp = str(tmp_path / 'grid.json')
    layout = _create_dummy_grid(grid_fp)

    harm_dpath = str(tmp_path / 'harmonized')
    run_dpath = os.path.join(harm_dpath, 'run_0001')
    os.makedirs(run_dpath, exist_ok=True)

    report_content = {
        'run_uid': 'uid_test_0001',
        'run_id': 'run_0001',
        'timestamp': '2026-09-18T00:00:00Z',
        'status': 'SUCCESS',
        'provenance': {},
        'harmonized_sources': {},
        'finalized_rasters': {
            'features': '/final/features.vrt',
            'labels': '/final/labels.vrt',
            'domain_landcover': '/final/domain_lc.tif',
        },
        'valid_mask_raster': '/final/valid_pixel_mask.vrt',
        'grid_id': layout.gid,
        'grid_fpath': grid_fp,
    }
    report_fp = os.path.join(run_dpath, 'harmonize_report.json')
    artifacts.Controller[contracts.HarmonizationReportSchema](
        report_fp
    ).persist(report_content)

    harm_paths = artifacts.HarmonizationPaths(harm_dpath)
    harm_paths.init_pipeline_folders()
    manifest_rec: contracts.HarmonizationRunRecord = {
        'run_uid': 'uid_test_0001',
        'run_id': 'run_0001',
        'run_folder': run_dpath,
        'status': 'SUCCESS',
        'timestamp': '2026-09-18T00:00:00Z',
    }
    artifacts.Controller[dict](harm_paths.runs_manifest).persist({
        'uid_test_0001': manifest_rec
    })

    ctx = ingest_context.build_ingestion_context(
        harm_paths, harmonization_run_id=1
    )

    assert ctx.grid.gid == layout.gid
    assert ctx.grid_fpath == grid_fp
    assert ctx.features == '/final/features.vrt'
    assert ctx.labels == '/final/labels.vrt'
    assert ctx.domains == {'domain_landcover': '/final/domain_lc.tif'}
    assert ctx.has_data is True
    assert ctx.run_uid == 'uid_test_0001'
    assert ctx.harmonization_run_id == 'run_0001'


def test_discover_runs_missing_manifest_raises(tmp_path):
    '''
    Given: Harmonization root without a runs manifest.
    When: `discover_successful_harmonization_runs` is invoked.
    Then: Raise ArtifactError.
    '''
    harm_paths = artifacts.HarmonizationPaths(str(tmp_path / 'nonexistent'))
    with pytest.raises(
        artifacts.ArtifactError,
        match='manifest does not exist'
    ):
        ingest_context.discover_successful_harmonization_runs(harm_paths)


def test_discover_runs_no_successful_runs_raises(tmp_path):
    '''
    Given: Harmonization runs manifest containing only failed runs.
    When: `discover_successful_harmonization_runs` is invoked.
    Then: Raise ArtifactError indicating no successful runs.
    '''
    harm_paths = artifacts.HarmonizationPaths(str(tmp_path / 'harm'))
    harm_paths.init_pipeline_folders()
    manifest_rec: contracts.HarmonizationRunRecord = {
        'run_uid': 'uid_fail_0001',
        'run_id': 'run_0001',
        'run_folder': '/fake/run_0001',
        'status': 'FAILED',
        'timestamp': '2026-09-18T00:00:00Z',
    }
    artifacts.Controller[dict](harm_paths.runs_manifest).persist({
        'uid_fail_0001': manifest_rec
    })

    with pytest.raises(
        artifacts.ArtifactError,
        match='No successful harmonization runs found'
    ):
        ingest_context.discover_successful_harmonization_runs(harm_paths)


def test_resolve_harmonization_run_by_uid_and_latest():
    '''
    Given: Successful runs manifest mapping.
    When: Resolving by target UID, index, run_id, and None.
    Then: Return matching records accordingly.
    '''
    runs = {
        'uid_1': {
            'run_uid': 'uid_1',
            'run_id': 'run_0001',
            'run_folder': '/data/run_0001',
            'status': 'SUCCESS',
            'timestamp': '2026-09-18T00:00:00Z',
        },
        'uid_2': {
            'run_uid': 'uid_2',
            'run_id': 'run_0002',
            'run_folder': '/data/run_0002',
            'status': 'SUCCESS',
            'timestamp': '2026-09-19T00:00:00Z',
        },
    }

    # latest when target is None
    latest = ingest_context.resolve_harmonization_run(runs, None)
    assert latest['run_uid'] == 'uid_2'

    # target by UID
    by_uid = ingest_context.resolve_harmonization_run(runs, 'uid_1')
    assert by_uid['run_id'] == 'run_0001'

    # target by int index
    by_idx = ingest_context.resolve_harmonization_run(runs, 2)
    assert by_idx['run_uid'] == 'uid_2'

    # target by run_id string
    by_name = ingest_context.resolve_harmonization_run(runs, 'run_0001')
    assert by_name['run_uid'] == 'uid_1'

    # target not found raises
    with pytest.raises(artifacts.ArtifactError, match='not found'):
        ingest_context.resolve_harmonization_run(runs, 'unknown_uid')


def test_discover_ingested_harmonization_uids(tmp_path):
    '''
    Given: Ingestion runs manifest with SUCCESS and FAILED entries.
    When: `discover_ingested_harmonization_uids` is invoked.
    Then: Return only UIDs corresponding to SUCCESS runs.
    '''
    ingest_paths = artifacts.IngestionPaths(str(tmp_path / 'ingest'))
    # when manifest does not exist
    assert ingest_context.discover_ingested_harmonization_uids(
        ingest_paths
    ) == set()

    ingest_paths.init_pipeline_folders()
    manifest_data = {
        'ing_1': {
            'run_uid': 'ing_1',
            'run_id': 'run_0001',
            'harmonization_run_uid': 'harmonize_uid_1',
            'harmonization_run_id': 'run_0001',
            'status': 'SUCCESS',
            'timestamp': '2026-09-24T00:00:00Z',
            'run_folder': '/path/1',
        },
        'ing_2': {
            'run_uid': 'ing_2',
            'run_id': 'run_0002',
            'harmonization_run_uid': 'harmonize_uid_2',
            'harmonization_run_id': 'run_0002',
            'status': 'FAILED',
            'timestamp': '2026-09-24T01:00:00Z',
            'run_folder': '/path/2',
        },
    }
    artifacts.Controller[dict](ingest_paths.runs_manifest).persist(
        manifest_data
    )

    uids = ingest_context.discover_ingested_harmonization_uids(
        ingest_paths
    )
    assert uids == {'harmonize_uid_1'}


def test_resolve_pending_ingestion_batches(tmp_path):
    '''
    Given: Harmonization manifest with 2 runs, and 1 already ingested.
    When: `resolve_pending_ingestion_batches` is called under modes.
    Then: Correctly plan batches for pending, latest, and targeted.
    '''
    harm_paths = artifacts.HarmonizationPaths(str(tmp_path / 'harm'))
    harm_paths.init_pipeline_folders()
    ingest_paths = artifacts.IngestionPaths(str(tmp_path / 'ingest'))
    ingest_paths.init_pipeline_folders()

    harm_manifest = {
        'harm_1': {
            'run_uid': 'harm_1',
            'run_id': 'run_0001',
            'run_folder': str(tmp_path / 'harm' / 'run_0001'),
            'status': 'SUCCESS',
            'timestamp': '2026-09-24T00:00:00Z',
        },
        'harm_2': {
            'run_uid': 'harm_2',
            'run_id': 'run_0002',
            'run_folder': str(tmp_path / 'harm' / 'run_0002'),
            'status': 'SUCCESS',
            'timestamp': '2026-09-24T01:00:00Z',
        },
    }
    artifacts.Controller[dict](harm_paths.runs_manifest).persist(
        harm_manifest
    )

    # harm_1 has already been ingested
    ingest_manifest = {
        'ing_1': {
            'run_uid': 'ing_1',
            'run_id': 'run_0001',
            'harmonization_run_uid': 'harm_1',
            'harmonization_run_id': 'run_0001',
            'status': 'SUCCESS',
            'timestamp': '2026-09-24T00:00:00Z',
            'run_folder': str(tmp_path / 'ingest' / 'run_0001'),
        }
    }
    artifacts.Controller[dict](ingest_paths.runs_manifest).persist(
        ingest_manifest
    )

    # pending / auto mode (target=None) -> only harm_2
    pending = ingest_context.resolve_pending_ingestion_batches(
        harm_paths, ingest_paths, target=None
    )
    assert len(pending) == 1
    assert pending[0]['run_uid'] == 'harm_2'

    # pending mode with rebuild=True -> both runs
    rebuild_all = ingest_context.resolve_pending_ingestion_batches(
        harm_paths, ingest_paths, target=None, rebuild=True
    )
    assert len(rebuild_all) == 2

    # targeted already ingested without rebuild -> empty list
    already_done = ingest_context.resolve_pending_ingestion_batches(
        harm_paths, ingest_paths, target='run_0001', rebuild=False
    )
    assert already_done == []

    # targeted already ingested with rebuild -> [harm_1]
    rebuild_target = ingest_context.resolve_pending_ingestion_batches(
        harm_paths, ingest_paths, target='run_0001', rebuild=True
    )
    assert len(rebuild_target) == 1
    assert rebuild_target[0]['run_uid'] == 'harm_1'

    # targeted uningested -> [harm_2]
    target_new = ingest_context.resolve_pending_ingestion_batches(
        harm_paths, ingest_paths, target='run_0002'
    )
    assert len(target_new) == 1
    assert target_new[0]['run_uid'] == 'harm_2'

    # latest mode -> [harm_2]
    latest = ingest_context.resolve_pending_ingestion_batches(
        harm_paths, ingest_paths, target='latest'
    )
    assert len(latest) == 1
    assert latest[0]['run_uid'] == 'harm_2'
