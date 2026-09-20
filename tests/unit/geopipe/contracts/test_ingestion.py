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

'''Unit tests for ingestion contracts and report schemas.'''

# local imports
import landseg.geopipe.contracts as contracts


# ----- contract schemas tests
def test_domain_stats_contract():
    '''
    Given: Attributes required for domain layer statistics.
    When: Instantiating a `DomainStats` TypedDict.
    Then: All fields match the contract specification.
    '''
    stats: contracts.DomainStats = {
        'max_index': 10,
        'valid_coords_count': 100,
        'major_freq_mean': 0.75,
        'major_freq_min': 0.50,
        'pca_axes_n': 3,
        'explained_variance': 0.95,
    }
    assert stats['max_index'] == 10
    assert stats['valid_coords_count'] == 100
    assert stats['major_freq_mean'] == 0.75
    assert stats['major_freq_min'] == 0.50
    assert stats['pca_axes_n'] == 3
    assert stats['explained_variance'] == 0.95


def test_domain_map_report_contract():
    '''
    Given: Attributes required for a domain map report.
    When: Instantiating a `DomainMapReport` TypedDict.
    Then: All fields and nested `DomainStats` match specification.
    '''
    stats: contracts.DomainStats = {
        'max_index': 5,
        'valid_coords_count': 50,
        'major_freq_mean': 0.8,
        'major_freq_min': 0.6,
        'pca_axes_n': 2,
        'explained_variance': 0.9,
    }
    report: contracts.DomainMapReport = {
        'name': 'landcover',
        'status': 'created',
        'input_filepath': '/data/lc.tif',
        'domain_filepath': '/data/lc_domain.json',
        'tiles_filepath': '/data/lc_tiles.json',
        'duration_sec': 12.34,
        'stats': stats,
    }
    assert report['name'] == 'landcover'
    assert report['status'] == 'created'
    assert report['stats'] is not None
    assert report['stats']['max_index'] == 5


def test_block_stats_contract():
    '''
    Given: Attributes required for data block generation stats.
    When: Instantiating a `BlockStats` TypedDict.
    Then: All metrics conform to contract schema.
    '''
    stats: contracts.BlockStats = {
        'shared_raster_windows': 100,
        'expected_shape_windows': 95,
        'blocks_on_disk_before': 10,
        'blocks_to_process': 85,
        'damaged_blocks_removed': 0,
        'blocks_created': 85,
    }
    assert stats['shared_raster_windows'] == 100
    assert stats['expected_shape_windows'] == 95
    assert stats['blocks_created'] == 85


def test_manifest_stats_contract():
    '''
    Given: Attributes required for catalog and schema updates.
    When: Instantiating a `ManifestStats` TypedDict.
    Then: All fields match the contract specification.
    '''
    stats: contracts.ManifestStats = {
        'catalog_status': 'synced',
        'cataloged_blocks_count': 120,
        'catalog_updated': True,
        'schema_updated': False,
    }
    assert stats['catalog_status'] == 'synced'
    assert stats['cataloged_blocks_count'] == 120
    assert stats['catalog_updated'] is True
    assert stats['schema_updated'] is False


def test_data_blocks_report_contract():
    '''
    Given: Attributes for data block execution report.
    When: Instantiating a `DataBlocksReport` TypedDict.
    Then: All fields conform to the contract.
    '''
    block_stats: contracts.BlockStats = {
        'shared_raster_windows': 10,
        'expected_shape_windows': 10,
        'blocks_on_disk_before': 0,
        'blocks_to_process': 10,
        'damaged_blocks_removed': 0,
        'blocks_created': 10,
    }
    manifest_stats: contracts.ManifestStats = {
        'catalog_status': 'created',
        'cataloged_blocks_count': 10,
        'catalog_updated': True,
        'schema_updated': True,
    }
    report: contracts.DataBlocksReport = {
        'image_filepath': '/data/img.tif',
        'label_filepath': '/data/lbl.tif',
        'duration_sec': 5.67,
        'stats': block_stats,
        'manifest': manifest_stats,
    }
    assert report['image_filepath'] == '/data/img.tif'
    assert report['label_filepath'] == '/data/lbl.tif'
    assert report['stats'] is not None
    assert report['stats']['blocks_created'] == 10
    assert report['manifest'] is not None
    assert report['manifest']['catalog_updated'] is True


def test_ingest_report_schema_contract():
    '''
    Given: A full set of ingestion pipeline outputs.
    When: Instantiating an `IngestReportSchema` TypedDict.
    Then: All root fields and nested reports conform to contract.
    '''
    domain_report: contracts.DomainMapReport = {
        'name': 'forest',
        'status': 'loaded',
        'input_filepath': '/data/forest.tif',
        'domain_filepath': '/data/forest_domain.json',
        'tiles_filepath': '/data/forest_tiles.json',
        'duration_sec': 1.2,
        'stats': None,
    }
    schema: contracts.IngestReportSchema = {
        'run_id': 'ingest_run_01',
        'timestamp': '2026-09-18T00:00:00Z',
        'status': 'SUCCESS',
        'domain_maps': [domain_report],
        'data_blocks': None,
    }
    assert schema['run_id'] == 'ingest_run_01'
    assert schema['status'] == 'SUCCESS'
    assert len(schema['domain_maps']) == 1
    assert schema['domain_maps'][0]['name'] == 'forest'
    assert schema['data_blocks'] is None
