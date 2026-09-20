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

'''Unit tests for geopipe runtime DataSpecs builder (factory.py).'''

# standard imports
import typing
# third-party imports
import pytest
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.factory as factory


# ----- `_get_heads` tests
def test_get_heads_with_prepared_heads():
    '''
    Given: Prepared schema containing a `heads` reclassification section.
    When: Populating `core.Heads` via `_get_heads`.
    Then: Correctly map `head_parent` and `head_parent_cls` for heads.
    '''
    data_schema: typing.Any = {
        'labels': {
            'label_taxonomy': {},
        }
    }
    prepared_schema: typing.Any = {
        'label_stats': {
            'landcover': [100, 200, 300],
            'landcover_group': [300, 300],
            'landcover_VEG': [100, 200],
            'original': [600],
        },
        'heads': {
            'head_names': [
                'landcover',
                'landcover_group',
                'landcover_VEG',
            ],
            'head_parent': {
                'landcover': None,
                'landcover_group': None,
                'landcover_VEG': 'landcover_group',
            },
            'head_parent_cls': {
                'landcover': None,
                'landcover_group': None,
                'landcover_VEG': 1,
            },
            'num_classes': {
                'landcover': 3,
                'landcover_group': 2,
                'landcover_VEG': 2,
            },
            'class_names': {
                'landcover': ['c1', 'c2', 'c3'],
                'landcover_group': ['g1', 'g2'],
                'landcover_VEG': ['c1', 'c2'],
            },
            'ignore_classes': {
                'landcover': [255],
                'landcover_group': [255],
                'landcover_VEG': [255],
            },
        }
    }

    heads = factory._get_heads(data_schema, prepared_schema)

    # verify class counts exclude 'original'
    assert set(heads.class_counts.keys()) == {
        'landcover',
        'landcover_group',
        'landcover_VEG',
    }

    # verify parent relationships populated from prepared heads
    assert heads.head_parent == {
        'landcover': None,
        'landcover_group': None,
        'landcover_VEG': 'landcover_group',
    }
    assert heads.head_parent_cls == {
        'landcover': None,
        'landcover_group': None,
        'landcover_VEG': 1,
    }


def test_get_heads_without_prepared_heads_fallback():
    '''
    Given: Prepared schema lacking a `heads` section (legacy schema).
    When: Populating `core.Heads` via `_get_heads`.
    Then: Fall back to None for all head parent mappings.
    '''
    data_schema: typing.Any = {
        'labels': {
            'label_taxonomy': {},
        }
    }
    prepared_schema: typing.Any = {
        'label_stats': {
            'landcover': [100, 200],
            'original': [300],
        },
    }

    heads = factory._get_heads(data_schema, prepared_schema)

    assert heads.head_parent == {'landcover': None}
    assert heads.head_parent_cls == {'landcover': None}


# ----- `build_dataspec` tests
def test_build_dataspec(mocker):
    '''
    Given: Mocked controllers for data schema and prepared schema.
    When: Assembling runtime `DataSpecs` via `build_dataspec`.
    Then: Return populated `DataSpecs` with resolved parent hierarchy.
    '''
    mock_data_schema = {
        'dataset': {'name': 'test_dataset'},
        'io_conventions': {
            'dtypes': {'image': 'float32', 'label': 'int32'},
            'image_band_map': {'red': 0, 'green': 1},
            'ignore_index': 255,
        },
        'tensor_shapes': {
            'image': {'C': 2, 'H': 64, 'W': 64, 'shape': [2, 64, 64]},
            'label': {'L': 1, 'H': 64, 'W': 64, 'shape': [1, 64, 64]},
        },
        'labels': {
            'label_class_color_map': None,
            'label_taxonomy': {},
        },
    }

    mock_prepared_schema = {
        'image_array_key': 'image',
        'label_array_key': 'label',
        'train_blocks': {'row_000000_col_000000': 'path/to/b1.npz'},
        'val_blocks': {'row_000000_col_000064': 'path/to/b2.npz'},
        'test_blocks': {'row_000064_col_000000': 'path/to/b3.npz'},
        'label_stats': {
            'head_base': [50, 50],
            'head_child': [50],
        },
        'heads': {
            'head_names': ['head_base', 'head_child'],
            'head_parent': {
                'head_base': None,
                'head_child': 'head_base',
            },
            'head_parent_cls': {
                'head_base': None,
                'head_child': 1,
            },
            'num_classes': {'head_base': 2, 'head_child': 1},
            'class_names': {'head_base': ['a', 'b'], 'head_child': ['a']},
            'ignore_classes': {'head_base': [255], 'head_child': [255]},
        },
    }

    # mock artifact controller loaders
    mocker.patch(
        'landseg.artifacts.Controller.load_json_or_fail',
        side_effect=lambda fpath: mocker.Mock(
            fetch=mocker.Mock(
                return_value=(
                    mock_data_schema
                    if 'dev.schema' in fpath
                    else mock_prepared_schema
                )
            )
        )
    )

    mock_paths = mocker.Mock()
    mock_paths.data_ingestion.data_blocks.schema = 'dev.schema.json'
    mock_paths.data_preparation.schema = 'prep.schema.json'
    mock_paths.knowledge = None

    specs = factory.build_dataspec(mock_paths, mode='default')

    assert specs.name == 'test_dataset'
    assert specs.mode == 'default'
    assert specs.heads.head_parent == {
        'head_base': None,
        'head_child': 'head_base',
    }
    assert specs.heads.head_parent_cls == {
        'head_base': None,
        'head_child': 1,
    }
