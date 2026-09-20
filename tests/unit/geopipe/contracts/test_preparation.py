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

'''Unit tests for dataset preparation contracts and report schemas.'''

# local imports
import landseg.geopipe.contracts as contracts


# ----- contract schemas tests
def test_blocks_partition_contract():
    '''
    Given: File path mappings for train, validation, and test splits.
    When: Instantiating a `BlocksPartition` TypedDict.
    Then: All partition mappings match the contract specification.
    '''
    partition: contracts.BlocksPartition = {
        'train': {'blk_1': '/data/blk_1.npz'},
        'val': {'blk_2': '/data/blk_2.npz'},
        'test': {'blk_3': '/data/blk_3.npz'},
    }
    assert partition['train']['blk_1'] == '/data/blk_1.npz'
    assert partition['val']['blk_2'] == '/data/blk_2.npz'
    assert partition['test']['blk_3'] == '/data/blk_3.npz'


def test_image_band_stats_contract():
    '''
    Given: Attributes required for per-band statistical summary.
    When: Instantiating an `ImageBandStats` TypedDict.
    Then: All metrics conform to the contract schema.
    '''
    stats: contracts.ImageBandStats = {
        'total_count': 1000,
        'current_mean': 128.5,
        'accum_m2': 45000.0,
        'std': 6.71,
    }
    assert stats['total_count'] == 1000
    assert stats['current_mean'] == 128.5
    assert stats['accum_m2'] == 45000.0
    assert stats['std'] == 6.71


def test_target_heads_schema_contract():
    '''
    Given: Attributes defining multi-head targets hierarchy.
    When: Instantiating a `TargetHeadsSchema` TypedDict.
    Then: All fields match the contract specification.
    '''
    heads: contracts.TargetHeadsSchema = {
        'head_names': ['landcover', 'canopy'],
        'head_parent': {'landcover': None, 'canopy': 'landcover'},
        'head_parent_cls': {'landcover': None, 'canopy': 1},
        'num_classes': {'landcover': 4, 'canopy': 2},
        'class_names': {
            'landcover': ['WAT', 'FOR', 'WET', 'UCL'],
            'canopy': ['LOW', 'HIGH'],
        },
        'ignore_classes': {'landcover': [255], 'canopy': [255]},
    }
    assert heads['head_names'] == ['landcover', 'canopy']
    assert heads['num_classes']['landcover'] == 4
    assert heads['head_parent']['canopy'] == 'landcover'


def test_prepared_schema_contract():
    '''
    Given: Complete dataset preparation metadata and metrics.
    When: Instantiating a `PreparedSchema` TypedDict.
    Then: Conforms to schema ID and includes nested statistics.
    '''
    stats: contracts.ImageBandStats = {
        'total_count': 500,
        'current_mean': 10.0,
        'accum_m2': 100.0,
        'std': 0.45,
    }
    schema: contracts.PreparedSchema = {
        'schema_version': contracts.PREPARED_SCHEMA_ID,
        'creation_time': '2026-09-18T00:00:00Z',
        'artifacts': {'splits': '/data/splits.json'},
        'checksums': {'splits': 'sha256-hash'},
        'train_blocks': {'b1': '/b1.npz'},
        'val_blocks': {'b2': '/b2.npz'},
        'test_blocks': {'b3': '/b3.npz'},
        'label_stats': {'landcover': [100, 200, 300]},
        'image_stats': {'band_0': stats},
        'image_array_key': 'image',
        'label_array_key': 'label',
    }
    assert schema['schema_version'] == contracts.PREPARED_SCHEMA_ID
    assert schema['image_array_key'] == 'image'
    assert schema['label_stats']['landcover'] == [100, 200, 300]


def test_partition_summary_contract():
    '''
    Given: Details for original splits and hydration results.
    When: Instantiating a `PartitionSummary` TypedDict.
    Then: Correctly capture counts and hydration stats.
    '''
    summary: contracts.PartitionSummary = {
        'original_splits': {
            'training': {
                'num_of_blocks': 80,
                'class_count': [40, 40],
                'class_distribution': [0.5, 0.5],
            },
            'validation': {
                'num_of_blocks': 10,
                'class_count': [5, 5],
                'class_distribution': [0.5, 0.5],
            },
            'testing': {
                'num_of_blocks': 10,
                'class_count': [5, 5],
                'class_distribution': [0.5, 0.5],
            },
        },
        'hydration': {
            'performed': True,
            'focal_head': 'landcover',
            'stop_reason': 'target_reached',
            'n_training_blocks': 90,
            'n_training_blocks_change': 10,
            'hydrated_class_count': [45, 45],
            'class_count_change': [5, 5],
            'hydrated_class_distribution': [0.5, 0.5],
            'class_distribution_change': [0.0, 0.0],
        },
    }
    assert summary['hydration']['performed'] is True
    assert summary['hydration']['n_training_blocks'] == 90
    assert summary['original_splits']['training']['num_of_blocks'] == 80


def test_data_partition_report_contract():
    '''
    Given: Attributes for data partitioning report.
    When: Instantiating a `DataPartitionReport` TypedDict.
    Then: Fields conform to the contract.
    '''
    report: contracts.DataPartitionReport = {
        'status': 'created',
        'duration_sec': 4.25,
    }
    assert report['status'] == 'created'
    assert report['duration_sec'] == 4.25


def test_normalization_report_contract():
    '''
    Given: Attributes for block normalization execution report.
    When: Instantiating a `NormalizationReport` TypedDict.
    Then: Fields conform to the contract.
    '''
    report: contracts.NormalizationReport = {
        'status': 'created',
        'duration_sec': 12.5,
        'unwanted_blocks_removed': 2,
        'rebuild': False,
        'stats_filepath': '/data/stats.json',
    }
    assert report['status'] == 'created'
    assert report['unwanted_blocks_removed'] == 2


def test_schema_report_contract():
    '''
    Given: Attributes for schema generation execution report.
    When: Instantiating a `SchemaReport` TypedDict.
    Then: Fields conform to the contract.
    '''
    report: contracts.SchemaReport = {
        'status': 'loaded',
        'duration_sec': 0.75,
        'schema_filepath': '/data/schema.json',
        'classes_mapped': ['landcover'],
    }
    assert report['status'] == 'loaded'
    assert report['classes_mapped'] == ['landcover']


def test_preparation_report_schema_contract():
    '''
    Given: Full pipeline stage reports for preparation.
    When: Instantiating a `PreparationReportSchema` TypedDict.
    Then: Root fields and sub-stage reports conform to contract.
    '''
    part_report: contracts.DataPartitionReport = {
        'status': 'created',
        'duration_sec': 1.0,
    }
    norm_report: contracts.NormalizationReport = {
        'status': 'created',
        'duration_sec': 2.0,
        'unwanted_blocks_removed': 0,
        'rebuild': False,
        'stats_filepath': '/stats.json',
    }
    schema_report: contracts.SchemaReport = {
        'status': 'created',
        'duration_sec': 0.5,
        'schema_filepath': '/schema.json',
        'classes_mapped': ['c1'],
    }
    prep_schema: contracts.PreparationReportSchema = {
        'run_id': 'prep_run_01',
        'timestamp': '2026-09-18T00:00:00Z',
        'status': 'SUCCESS',
        'data_partition': part_report,
        'normalization': norm_report,
        'schema': schema_report,
    }
    assert prep_schema['run_id'] == 'prep_run_01'
    assert prep_schema['status'] == 'SUCCESS'
    assert prep_schema['data_partition'] is not None
    assert prep_schema['normalization'] is not None
    assert prep_schema['schema'] is not None
