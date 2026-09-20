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

'''Unit tests for world grid contracts and report schemas.'''

# local imports
import landseg.geopipe.contracts as contracts


# ----- contract schemas tests
def test_world_grid_report_contract():
    '''
    Given: Attributes required for a world grid summary report.
    When: Instantiating a `WorldGridReport`.
    Then: All fields match the contract specification.
    '''
    report: contracts.WorldGridReport = {
        'grid_fpath': '/path/to/grid.json',
        'grid_id': 'grid_row_256_128_col_256_128',
        'crs': 'EPSG:3161',
        'pixel_size': (20.0, 20.0),
        'tile_size': (256, 256),
        'tile_overlap': (128, 128),
    }
    assert report['grid_id'] == 'grid_row_256_128_col_256_128'
    assert report['pixel_size'] == (20.0, 20.0)
    assert report['tile_size'] == (256, 256)
    assert report['tile_overlap'] == (128, 128)


def test_grid_report_schema_contract():
    '''
    Given: Attributes required for the root world grid pipeline report.
    When: Instantiating a `GridReportSchema`.
    Then: All fields match the execution report specification.
    '''
    grid_report: contracts.WorldGridReport = {
        'grid_fpath': '/path/to/grid.json',
        'grid_id': 'grid_row_256_128_col_256_128',
        'crs': 'EPSG:3161',
        'pixel_size': (20.0, 20.0),
        'tile_size': (256, 256),
        'tile_overlap': (128, 128),
    }
    report: contracts.GridReportSchema = {
        'run_id': 'world-grid',
        'timestamp': '2026-09-18T00:00:00',
        'status': 'SUCCESS',
        'grid': grid_report,
        'total_tiles': 42,
    }
    assert report['run_id'] == 'world-grid'
    assert report['status'] == 'SUCCESS'
    assert report['grid']['grid_id'] == 'grid_row_256_128_col_256_128'
    assert report['total_tiles'] == 42
