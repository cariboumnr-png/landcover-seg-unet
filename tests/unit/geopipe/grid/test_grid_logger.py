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

'''Unit tests for `GridLogger` and grid report lifecycle utilities.'''

# standard imports
import json
import os
# local imports
import landseg.geopipe.contracts as contracts
import landseg.geopipe.grid as grid


# ----- `GridLogger` lifecycle tests
def test_grid_logger_summary_lifecycle(tmp_path):
    '''
    Given: A GridLogger initialized with report artifact path.
    When: Recording world grid metrics and closing.
    Then: Persists structured grid_report.json to output directory.
    '''
    report_file = os.path.join(str(tmp_path), 'grid_report.json')

    logger = grid.GridLogger(
        name='test_grid',
        log_file=report_file,
        enable_file_log=False
    )
    logger.init_summary(run_id='world-grid')

    grid_report: contracts.WorldGridReport = {
        'grid_fpath': '/path/to/grid.json',
        'grid_id': 'grid_row_256_128_col_256_128',
        'crs': 'EPSG:3161',
        'pixel_size': (20.0, 20.0),
        'tile_size': (256, 256),
        'tile_overlap': (128, 128),
    }
    logger.set_grid_report(grid_report, total_tiles=42)
    logger.close()

    assert os.path.exists(report_file)
    with open(report_file, 'r', encoding='utf-8') as f:
        saved_report = json.load(f)

    assert saved_report['status'] == 'SUCCESS'
    assert saved_report['run_id'] == 'world-grid'
    assert saved_report['total_tiles'] == 42
    assert saved_report['grid']['grid_id'] == (
        'grid_row_256_128_col_256_128'
    )


def test_grid_logger_failure_status(tmp_path):
    '''
    Given: A GridLogger tracking a failed execution.
    When: Setting status to 'FAILED' and closing.
    Then: Persisted report records 'FAILED' status.
    '''
    report_file = os.path.join(str(tmp_path), 'grid_report.json')

    logger = grid.GridLogger(
        name='test_grid',
        log_file=report_file,
        enable_file_log=False
    )
    logger.init_summary(run_id='world-grid')
    logger.set_summary_status('FAILED')
    logger.close()

    with open(report_file, 'r', encoding='utf-8') as f:
        saved_report = json.load(f)

    assert saved_report['status'] == 'FAILED'
