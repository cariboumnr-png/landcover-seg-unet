# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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

# pylint: disable=missing-function-docstring

'''Fixtures for testing `landseg.core` module.'''

# third-party imports
import pytest
# local imports
import landseg.core as core

@pytest.fixture
def dataspecs():
    return core.DataSpecs(
        name="test_dataset",
        mode="default",
        meta=core.Meta(
            blk_bytes=1024,
            test_blks_grid=(2, 2),
            label_color_map=None,
            image_specs=core.Meta.Image(
                num_channels=4,
                height_width=256,
                array_key='image',
                band_map={'red': 0, 'green': 1, 'blue': 2, 'dem': 3}
            ),
            label_specs=core.Meta.Label(
                ignore_index=255,
                array_key='label'
            )
        ),
        heads=core.Heads(
            class_counts={'head1': [100, 200]},
            logits_adjust={'head1': [0.2, 0.1]},
            head_parent={'head1': None},
            head_parent_cls={'head1': None},
        ),
        splits=core.Splits(
            train={'block_1': 'file//path//1'},
            val={'block_2': 'file//path//2'},
            test={'block_3': 'file//path//3'},
        ),
        domains=core.Domains(
        train=core.Domains.Dom(
                ids_domain={'block_1': 1},
                vec_domain={'block_1': [0.1, 0.2]}
            ),
            val=core.Domains.Dom(
                ids_domain={'block_2': 2},
                vec_domain={'block_2': [0.3, 0.4]}
            ),
            test=core.Domains.Dom(
                ids_domain={'block_3': 3},
                vec_domain={'block_3': [0.5, 0.6]}
            ),
            ids_num=3,
            vec_dim=2,
        ),
    )
