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
Unit tests for `PreparationLogger` structured logging and summary
persistence.
'''

# standard imports
import json
import os
# local imports
import landseg.geopipe.contracts as contracts
import landseg.geopipe.prepare as prepare


# ----- test cases
def test_preparation_logger_summary_lifecycle(tmp_path):
    '''
    Given: A `PreparationLogger` initialized with output report path.
    When: Recording partition, normalization, schema reports, and
        closing.
    Then: Persist structured preparation summary JSON to target path.
    '''
    out_dpath = str(tmp_path / 'prep_out')
    os.makedirs(out_dpath, exist_ok=True)
    report_file = os.path.join(out_dpath, 'prepare_report.json')

    logger = prepare.PreparationLogger(
        name='test_prep',
        log_file=report_file,
        enable_file_log=False,
    )
    logger.init_summary(run_id='run_prep_001')

    part_report: contracts.DataPartitionReport = {
        'status': 'created',
        'duration_sec': 5.2,
    }
    logger.set_data_partition_report(part_report)

    norm_report: contracts.NormalizationReport = {
        'status': 'created',
        'duration_sec': 14.1,
        'unwanted_blocks_removed': 0,
        'rebuild': False,
        'stats_filepath': '/data/stats.json',
    }
    logger.set_normalization_report(norm_report)

    schema_report: contracts.SchemaReport = {
        'status': 'created',
        'duration_sec': 1.8,
        'schema_filepath': '/data/schema.json',
        'classes_mapped': ['landcover'],
    }
    logger.set_schema_report(schema_report)
    logger.set_summary_status('SUCCESS')
    logger.close()

    assert os.path.exists(report_file)
    with open(report_file, 'r', encoding='utf-8') as f:
        saved_report = json.load(f)

    assert saved_report['run_id'] == 'run_prep_001'
    assert saved_report['status'] == 'SUCCESS'
    assert saved_report['data_partition'] is not None
    assert saved_report['data_partition']['status'] == 'created'
    assert saved_report['normalization'] is not None
    assert saved_report['normalization']['stats_filepath'] == (
        '/data/stats.json'
    )
    assert saved_report['schema'] is not None
    assert saved_report['schema']['classes_mapped'] == ['landcover']


def test_preparation_logger_uninitialized_summary_noop(tmp_path):
    '''
    Given: A `PreparationLogger` without calling `init_summary`.
    When: Setting reports, updating status, and closing.
    Then: No errors raised and summary remains None without output file.
    '''
    report_file = str(tmp_path / 'uninit_report.json')
    logger = prepare.PreparationLogger(
        name='test_uninit',
        log_file=report_file,
        enable_file_log=False,
    )
    part_report: contracts.DataPartitionReport = {
        'status': 'loaded',
        'duration_sec': 0.1,
    }
    logger.set_data_partition_report(part_report)
    logger.set_summary_status('FAILED')
    assert logger.summary is None
    logger.close()
    assert not os.path.exists(report_file)


def test_preparation_logger_custom_timestamp(tmp_path):
    '''
    Given: An explicit ISO-formatted timestamp.
    When: Initializing summary with the custom timestamp.
    Then: Persist the exact timestamp in the summary report.
    '''
    report_file = str(tmp_path / 'time_report.json')
    logger = prepare.PreparationLogger(
        name='test_timestamp',
        log_file=report_file,
        enable_file_log=False,
    )
    logger.init_summary(
        run_id='run_custom_time',
        timestamp='2026-09-18T18:00:00Z',
    )
    logger.close()

    assert os.path.exists(report_file)
    with open(report_file, 'r', encoding='utf-8') as f:
        saved_report = json.load(f)
    assert saved_report['timestamp'] == '2026-09-18T18:00:00Z'
