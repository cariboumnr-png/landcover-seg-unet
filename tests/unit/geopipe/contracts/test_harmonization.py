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

'''Unit tests for harmonization contracts and report schemas.'''

# local imports
import landseg.geopipe.contracts as contracts


# ----- contract schemas tests
def test_provenance_record_contract():
    '''
    Given: Attributes required for a source raster provenance record.
    When: Instantiating a `ProvenanceRecord`.
    Then: All fields match the contract specification.
    '''
    record: contracts.ProvenanceRecord = {
        'path': '/path/to/raster.tif',
        'size_bytes': 2048,
        'mtime': 1700000000.0,
    }
    assert record['path'] == '/path/to/raster.tif'
    assert record['size_bytes'] == 2048
    assert record['mtime'] == 1700000000.0


def test_harmonization_report_schema_contract():
    '''
    Given: A full set of harmonization execution outputs.
    When: Instantiating a `HarmonizationReportSchema`.
    Then: All fields and nested schemas conform to the contract.
    '''
    provenance: dict[str, contracts.ProvenanceRecord] = {
        's2': {
            'path': '/data/s2.tif',
            'size_bytes': 4096,
            'mtime': 1700000000.0,
        }
    }
    schema: contracts.HarmonizationReportSchema = {
        'run_id': 'run_42',
        'timestamp': '2026-09-17T00:00:00Z',
        'status': 'SUCCESS',
        'provenance': provenance,
        'harmonized_sources': {'s2': '/warped/s2.vrt'},
        'finalized_rasters': {'features': '/final/features.vrt'},
        'valid_mask_raster': '/final/mask.tif',
        'grid_id': 'ontario_20m',
        'grid_fpath': '/path/to/grid.json',
    }
    assert schema['run_id'] == 'run_42'
    assert schema['status'] == 'SUCCESS'
    assert schema['grid_id'] == 'ontario_20m'
    assert schema['grid_fpath'] == '/path/to/grid.json'
