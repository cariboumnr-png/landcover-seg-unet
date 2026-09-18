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
Unit tests for `IngestionLogger` structured logging and summary
persistence.
'''

# standard imports
import json
import os
# local imports
import landseg.geopipe.contracts as contracts
import landseg.geopipe.ingest as ingest


# ----- test cases
def test_ingestion_logger_summary_lifecycle(tmp_path):
    '''
    Given: An `IngestionLogger` initialized with a report output path.
    When: Recording domain reports, data blocks report, and closing.
    Then: Persist structured ingest summary JSON to the target path.
    '''
    out_dpath = str(tmp_path / 'ingest_out')
    os.makedirs(out_dpath, exist_ok=True)
    report_file = os.path.join(out_dpath, 'ingest_report.json')

    logger = ingest.IngestionLogger(
        name='test_ingest',
        log_file=report_file,
        enable_file_log=False,
    )
    logger.init_summary(run_id='run_ingest_001')

    domain_report: contracts.DomainMapReport = {
        'name': 'wetlands',
        'status': 'created',
        'input_filepath': '/raw/wetlands.tif',
        'domain_filepath': '/domains/wetlands.json',
        'tiles_filepath': '/tiles/wetlands.json',
        'duration_sec': 3.14,
        'stats': None,
    }
    logger.add_domain_report(domain_report)

    data_blocks_report: contracts.DataBlocksReport = {
        'image_filepath': '/harmonized/features.vrt',
        'label_filepath': '/harmonized/labels.tif',
        'duration_sec': 10.5,
        'stats': None,
        'manifest': None,
    }
    logger.set_data_blocks_report(data_blocks_report)
    logger.set_summary_status('SUCCESS')
    logger.close()

    assert os.path.exists(report_file)
    with open(report_file, 'r', encoding='utf-8') as f:
        saved_report = json.load(f)

    assert saved_report['run_id'] == 'run_ingest_001'
    assert saved_report['status'] == 'SUCCESS'
    assert len(saved_report['domain_maps']) == 1
    assert saved_report['domain_maps'][0]['name'] == 'wetlands'
    assert saved_report['data_blocks'] is not None
    assert saved_report['data_blocks']['image_filepath'] == (
        '/harmonized/features.vrt'
    )


def test_ingestion_logger_uninitialized_summary_noop(tmp_path):
    '''
    Given: An `IngestionLogger` without calling `init_summary`.
    When: Adding reports, updating status, and closing.
    Then: No errors raised and summary remains None.
    '''
    report_file = str(tmp_path / 'uninit_report.json')
    logger = ingest.IngestionLogger(
        name='test_uninit',
        log_file=report_file,
        enable_file_log=False,
    )
    domain_report: contracts.DomainMapReport = {
        'name': 'test',
        'status': 'loaded',
        'input_filepath': 'in.tif',
        'domain_filepath': 'dom.json',
        'tiles_filepath': 'tiles.json',
        'duration_sec': 0.1,
        'stats': None,
    }
    logger.add_domain_report(domain_report)
    logger.set_summary_status('FAILED')
    assert logger.summary is None
    logger.close()
    assert not os.path.exists(report_file)


def test_ingestion_logger_custom_timestamp(tmp_path):
    '''
    Given: An explicit ISO-formatted timestamp.
    When: Initializing summary with the custom timestamp.
    Then: Persist the exact timestamp in the summary report.
    '''
    report_file = str(tmp_path / 'time_report.json')
    logger = ingest.IngestionLogger(
        name='test_timestamp',
        log_file=report_file,
        enable_file_log=False,
    )
    logger.init_summary(
        run_id='run_custom_time',
        timestamp='2026-09-18T12:00:00Z',
    )
    logger.close()

    assert os.path.exists(report_file)
    with open(report_file, 'r', encoding='utf-8') as f:
        saved_report = json.load(f)
    assert saved_report['timestamp'] == '2026-09-18T12:00:00Z'
