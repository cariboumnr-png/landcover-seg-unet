# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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

# pylint: disable=protected-access

'''Unit tests for geopipe runtime DataSpecs builder (factory.py).'''

# standard imports
import typing
# local imports
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
