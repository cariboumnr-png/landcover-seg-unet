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

'''Unit tests for geopipe domain-specific logger subclasses.'''

# standard imports
import json
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.contracts as contracts
import landseg.geopipe.core as geo_core
import landseg.geopipe.grid as grid
import landseg.geopipe.harmonize as harmonize
import landseg.geopipe.ingest as ingest
import landseg.geopipe.prepare as prepare


# ----- `GridLogger` tests
def test_grid_logger_lifecycle(tmp_path):
    '''
    Given: A GridLogger instance.
    When: Initializing summary, recording grid report, and closing.
    Then: Persist structured GridReportSchema to disk matching contract.
    '''
    report_fp = tmp_path / 'grid_report.json'
    logger = grid.GridLogger(
        name='grid_test',
        log_file=report_fp,
        enable_file_log=False,
    )
    logger.init_summary(run_id='world-grid')

    spec = geo_core.GridSpec(
        crs='EPSG:32617',
        origin=(0.0, 1000.0),
        pixel_size=(10.0, 10.0),
        tile_size=(256, 256),
        tile_stride=(128, 128),
        grid_extent=(2560.0, 2560.0),
    )
    world_grid = geo_core.GridLayout(spec)
    grid_fp = tmp_path / 'grid.json'

    logger.set_grid_report(world_grid, grid_fp)
    logger.set_status('SUCCESS')
    logger.close()

    assert report_fp.exists()
    status, loaded = geo_core.read_report(report_fp)
    assert status == 'SUCCESS'
    assert loaded['grid_id'] == world_grid.gid
    assert loaded['crs'] == 'EPSG:32617'
    assert tuple(loaded['tile_size']) == (256, 256)


# ----- `HarmonizationLogger` tests
def test_harmonization_logger_lifecycle(tmp_path):
    '''
    Given: A HarmonizationLogger instance.
    When: Populating provenance, sources, and closing.
    Then: Persist structured HarmonizationReportSchema to disk.
    '''
    report_fp = tmp_path / 'harm_report.json'
    logger = harmonize.HarmonizationLogger(
        name='harm_test',
        log_file=report_fp,
        enable_file_log=False,
    )
    logger.init_summary(run_id='run_0001')
    uid = logger.run_uid
    assert uid.startswith('harmonize_')

    dummy_src = tmp_path / 'source.tif'
    dummy_src.write_bytes(b'dummy content')

    logger.add_source_provenance('s2', dummy_src)
    logger.add_harmonized_source('s2_harm', tmp_path / 's2_harm.tif')
    logger.add_finalized_raster('features', tmp_path / 'features.vrt')
    logger.set_valid_mask_raster(tmp_path / 'mask.vrt')
    logger.set_grid_reference('grid_utm17', tmp_path / 'grid.json')
    logger.set_identity('hash_12345')
    logger.close()

    assert report_fp.exists()
    with open(report_fp, 'r', encoding='UTF-8') as f:
        data = json.load(f)

    assert data['run_uid'] == uid
    assert data['run_id'] == 'run_0001'
    assert data['fingerprint'] == 'hash_12345'
    assert data['grid_id'] == 'grid_utm17'
    assert 's2' in data['provenance']
    assert data['finalized_rasters']['features'].endswith('features.vrt')


def test_harmonization_logger_update_runs_manifest(tmp_path):
    '''
    Given: A HarmonizationLogger with initialized summary.
    When: Calling update_runs_manifest for multiple sequential runs.
    Then: Persist and append records into the runs manifest JSON.
    '''
    manifest_fp = tmp_path / 'harmonization_runs.json'
    run_dir = tmp_path / 'run_0001'
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = harmonize.HarmonizationLogger(
        name='harm_manifest_test',
        log_file=tmp_path / 'rep.json',
        enable_file_log=False,
    )
    logger.init_summary(run_id='run_0001')
    logger.set_identity('fp_001')

    logger.update_runs_manifest(
        manifest_fp,
        run_dir,
        dataset_manifest_fpath='manifest.yaml',
    )

    ctrl = artifacts.Controller[dict](manifest_fp)
    # second run with FAILED status
    run_dir_2 = tmp_path / 'run_0002'
    run_dir_2.mkdir(parents=True, exist_ok=True)
    logger_2 = harmonize.HarmonizationLogger(
        name='harm_manifest_test_2',
        log_file=tmp_path / 'rep_2.json',
        enable_file_log=False,
    )
    logger_2.init_summary(run_id='run_0002')
    logger_2.set_status('FAILED')
    uid_2 = logger_2.run_uid

    logger_2.update_runs_manifest(
        manifest_fp,
        run_dir_2,
        dataset_manifest_fpath='manifest.yaml',
    )

    manifest_data = ctrl.fetch()
    assert manifest_data is not None
    assert len(manifest_data) == 2
    assert manifest_data[uid_2]['run_id'] == 'run_0002'
    assert manifest_data[uid_2]['status'] == 'FAILED'


# ----- `IngestionLogger` tests
def test_ingestion_logger_lifecycle(tmp_path):
    '''
    Given: An IngestionLogger instance.
    When: Populating domain reports and data block stats, then closing.
    Then: Persist structured IngestReportSchema to disk.
    '''
    report_fp = tmp_path / 'ingest_report.json'
    logger = ingest.IngestionLogger(
        name='ingest_test',
        log_file=report_fp,
        enable_file_log=False,
    )
    logger.init_summary(run_id='run_0001')
    assert logger.run_uid.startswith('ingest_')
    assert logger.run_id == 'run_0001'

    logger.set_identity('ingest_hash_abc')
    logger.set_harmonization_reference(
        run_uid='harm_uid_001',
        run_id='run_0001',
    )

    domain_rep: contracts.DomainMapReport = {
        'name': 'ecodistrict',
        'status': 'created',
        'input_filepath': '/path/eco.tif',
        'domain_filepath': '/path/eco.json',
        'tiles_filepath': '/path/tiles.npz',
        'duration_sec': 1.5,
        'stats': None,
    }
    logger.add_domain(domain_rep)

    blocks_rep: contracts.DataBlocksReport = {
        'image_filepath': '/path/img.tif',
        'label_filepath': '/path/lbl.tif',
        'duration_sec': 2.0,
        'stats': None,
        'manifest': None,
    }
    logger.set_data_blocks(blocks_rep)
    logger.close()

    assert report_fp.exists()
    with open(report_fp, 'r', encoding='UTF-8') as f:
        data = json.load(f)

    assert data['run_id'] == 'run_0001'
    assert data['harmonization_run_uid'] == 'harm_uid_001'
    assert len(data['domain_maps']) == 1
    assert data['domain_maps'][0]['name'] == 'ecodistrict'
    assert data['data_blocks']['image_filepath'] == '/path/img.tif'


def test_ingestion_logger_update_runs_manifest(tmp_path):
    '''
    Given: An IngestionLogger with initialized summary.
    When: Calling update_runs_manifest.
    Then: Persist run record with harmonization lineage.
    '''
    manifest_fp = tmp_path / 'ingestion_runs.json'
    run_dir = tmp_path / 'run_0001'
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = ingest.IngestionLogger(
        name='ingest_manifest_test',
        log_file=tmp_path / 'rep.json',
        enable_file_log=False,
    )
    logger.init_summary(run_id='run_0001')
    logger.set_harmonization_reference(
        run_uid='harm_ref_123',
        run_id='run_0001',
    )
    logger.set_identity('fp_ingest_999')

    logger.update_runs_manifest(manifest_fp, run_dir)

    ctrl = artifacts.Controller[dict](manifest_fp)
    # second run with FAILED status
    run_dir_2 = tmp_path / 'run_0002'
    run_dir_2.mkdir(parents=True, exist_ok=True)
    logger_2 = ingest.IngestionLogger(
        name='ingest_manifest_test_2',
        log_file=tmp_path / 'rep_2.json',
        enable_file_log=False,
    )
    logger_2.init_summary(run_id='run_0002')
    logger_2.set_harmonization_reference(
        run_uid='harm_ref_456',
        run_id='run_0002',
    )
    logger_2.set_status('FAILED')
    uid_2 = logger_2.run_uid

    logger_2.update_runs_manifest(manifest_fp, run_dir_2)

    manifest_data = ctrl.fetch()
    assert manifest_data is not None
    assert len(manifest_data) == 2
    assert manifest_data[uid_2]['run_id'] == 'run_0002'
    assert manifest_data[uid_2]['status'] == 'FAILED'


# ----- `PreparationLogger` tests
def test_preparation_logger_lifecycle(tmp_path):
    '''
    Given: A PreparationLogger instance.
    When: Populating stage reports and closing.
    Then: Persist structured PreparationReportSchema to disk.
    '''
    report_fp = tmp_path / 'prepare_report.json'
    logger = prepare.PreparationLogger(
        name='prepare_test',
        log_file=report_fp,
        enable_file_log=False,
    )
    logger.init_summary(run_id='run_0001')

    partition_rep: contracts.DataPartitionReport = {
        'status': 'created',
        'duration_sec': 1.0,
    }
    logger.set_data_partition(partition_rep)

    norm_rep: contracts.NormalizationReport = {
        'status': 'created',
        'duration_sec': 1.5,
        'unwanted_blocks_removed': 0,
        'rebuild': True,
        'stats_filepath': '/path/stats.json',
    }
    logger.set_normalization(norm_rep)

    schema_rep: contracts.SchemaReport = {
        'status': 'created',
        'duration_sec': 0.5,
        'schema_filepath': '/path/schema.json',
        'classes_mapped': ['landcover', 'landcover_group'],
    }
    logger.set_schema(schema_rep)
    logger.close()

    assert report_fp.exists()
    with open(report_fp, 'r', encoding='UTF-8') as f:
        data = json.load(f)

    assert data['run_id'] == 'run_0001'
    assert data['status'] == 'SUCCESS'
    assert data['data_partition']['status'] == 'created'
    assert data['normalization']['stats_filepath'] == '/path/stats.json'
    assert data['schema']['classes_mapped'] == ['landcover', 'landcover_group']
