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

'''
Unit tests for ETL `HarmonizationLogger` structured logger and report
persistence.
'''

# standard imports
import json
import os
# local imports
import landseg.geopipe.contracts as contracts
import landseg.geopipe.harmonize as harmonize


# ----- test cases
def test_harmonization_logger_summary_lifecycle(tmp_path):
    '''
    Given: A HarmonizationLogger initialized with output path.
    When: Recording source outputs, composite path, and closing.
    Then: Persists structured harmonize_report.json to output_dpath.
    '''
    out_dpath = str(tmp_path / 'harmonize_out')
    os.makedirs(out_dpath, exist_ok=True)
    report_file = os.path.join(out_dpath, 'harmonize_report.json')

    logger = harmonize.HarmonizationLogger(
        name='test_harmonize',
        log_file=report_file,
        enable_file_log=False
    )
    logger.init_summary(run_id='run_0001')
    logger.add_harmonized_source('sentinel2', '/path/to/s2.tif')
    logger.add_finalized_raster('stacked', '/path/to/stacked.tif')
    logger.set_valid_mask_raster('/path/to/mask.tif')
    logger.set_summary_status('SUCCESS')

    logger.close()

    assert os.path.exists(report_file)
    with open(report_file, 'r', encoding='utf-8') as f:
        saved_report = json.load(f)

    assert saved_report['status'] == 'SUCCESS'
    assert saved_report['run_id'] == 'run_0001'
    assert saved_report['harmonized_sources']['sentinel2'] == os.path.abspath(
        '/path/to/s2.tif'
    )
    assert saved_report['finalized_rasters']['stacked'] == os.path.abspath(
        '/path/to/stacked.tif'
    )
    assert saved_report['valid_mask_raster'] == os.path.abspath(
        '/path/to/mask.tif'
    )


def test_harmonization_logger_add_provenance(tmp_path):
    '''
    Given: A source file on disk and an initialized HarmonizationLogger.
    When: Calling `add_source_provenance`.
    Then: Records file size_bytes, mtime, and absolute path in report
        summary.
    '''
    out_dpath = str(tmp_path / 'prov_out')
    os.makedirs(out_dpath, exist_ok=True)
    sample_file = tmp_path / 'sample_raw.tif'
    sample_file.write_bytes(b'dummy_content_bytes')

    report_file = os.path.join(out_dpath, 'harmonize_report.json')
    logger = harmonize.HarmonizationLogger(
        name='test_provenance',
        log_file=report_file,
        enable_file_log=False
    )
    logger.init_summary(run_id='run_0001')

    logger.add_source_provenance('sentinel2', str(sample_file))
    logger.close()

    with open(report_file, 'r', encoding='utf-8') as f:
        report = json.load(f)

    prov = report['provenance']['sentinel2']
    assert prov['size_bytes'] == len(b'dummy_content_bytes')
    assert 'mtime' in prov
    assert prov['path'] == os.path.abspath(str(sample_file))


def test_harmonization_logger_set_grid_reference(tmp_path):
    '''
    Given: An initialized HarmonizationLogger.
    When: Setting grid reference with grid ID and file path.
    Then: Preserves grid reference in report summary and persists it.
    '''
    report_file = os.path.join(str(tmp_path), 'harmonize_report.json')
    logger = harmonize.HarmonizationLogger(
        name='test_grid_ref',
        log_file=report_file,
        enable_file_log=False,
    )
    logger.init_summary(run_id='run_0001')
    logger.set_grid_reference('grid_row_256_col_256', '/path/to/grid.json')
    logger.close()

    with open(report_file, 'r', encoding='utf-8') as f:
        report = json.load(f)

    assert report['grid_id'] == 'grid_row_256_col_256'
    assert report['grid_fpath'] == os.path.abspath('/path/to/grid.json')


def test_harmonization_logger_schema_types():
    '''
    Given: Instantiated typed dict schemas `ProvenanceRecord` and
        `HarmonizationReportSchema`.
    When: Populating valid fields according to report schema.
    Then: Successfully create structured schema instances.
    '''
    prov: contracts.ProvenanceRecord = {
        'path': '/path/to/raster.tif',
        'size_bytes': 1024,
        'mtime': 123456.78,
    }
    summary: contracts.HarmonizationReportSchema = {
        'run_id': 'run_0001',
        'timestamp': '2026-01-01T00:00:00Z',
        'status': 'SUCCESS',
        'provenance': {'sentinel2': prov},
        'harmonized_sources': {'sentinel2': '/path/to/s2.tif'},
        'finalized_rasters': {'stacked': '/path/to/stacked.tif'},
        'valid_mask_raster': '/path/to/mask.tif',
        'grid_id': 'grid_row_256_col_256',
        'grid_fpath': '/path/to/grid.json',
    }
    assert summary['status'] == 'SUCCESS'
    assert summary['provenance']['sentinel2']['size_bytes'] == 1024
    assert summary['grid_id'] == 'grid_row_256_col_256'
    assert summary['grid_fpath'] == '/path/to/grid.json'
