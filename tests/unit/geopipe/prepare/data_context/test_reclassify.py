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

'''Unit tests for label reclassification stack logic.'''

# standard imports
import pytest
# third-party imports
import numpy
# local imports
import landseg.geopipe.prepare.data_context.reclassify as reclassify


# ----- `reclassify_label_stack` tests
def test_reclassify_label_stack_2d():
    '''
    Given: Raw 2D label array and active target reclassification.
    When: Running reclassify_label_stack.
    Then: Construct base layer, child slices, and grouping layer.
    '''
    raw_arr = numpy.array([
        [1, 2],
        [3, 255]
    ], dtype=numpy.uint8)

    reclass_cfg = {
        'landcover': {
            'reclass': {'1': [1, 2], '2': [3]},
            'reclass_name': {'1': 'VEG', '2': 'WAT'},
        }
    }

    stack = reclassify.reclassify_label_stack(
        [raw_arr], ['landcover'], reclass_cfg, ignore_index=255
    )
    # expected 4 layers: base (1..3), child 1 (1..2 reindexed to 1, 2),
    # child 2 (3 reindexed to 1), and group layer (1, 2)
    assert stack.shape == (4, 2, 2)
    # base layer
    assert numpy.array_equal(stack[0], raw_arr)
    # child 1 (classes 1 and 2)
    assert stack[1, 0, 0] == 1
    assert stack[1, 0, 1] == 2
    assert stack[1, 1, 0] == 255
    # child 2 (class 3 reindexed to 1)
    assert stack[2, 0, 0] == 255
    assert stack[2, 1, 0] == 1
    # group layer (1, 2)
    assert stack[3, 0, 0] == 1
    assert stack[3, 0, 1] == 1
    assert stack[3, 1, 0] == 2
    assert stack[3, 1, 1] == 255


def test_reclassify_label_stack_3d_and_pass_through():
    '''
    Given: 3D label array without reclassification configured.
    When: Running reclassify_label_stack.
    Then: Return unchanged layers in the stack.
    '''
    raw_arr = numpy.zeros((1, 4, 4), dtype=numpy.uint8)
    stack = reclassify.reclassify_label_stack(
        raw_arr, ['landcover'], {}, ignore_index=255
    )
    assert stack.shape == (1, 4, 4)
    assert numpy.array_equal(stack, raw_arr)


def test_reclassify_label_stack_invalid_dim():
    '''
    Given: 1D label array with unsupported dimension.
    When: Running reclassify_label_stack.
    Then: Raise ValueError.
    '''
    raw_arr = numpy.zeros((10,), dtype=numpy.uint8)
    with pytest.raises(ValueError, match='Expected 2D or 3D label array'):
        reclassify.reclassify_label_stack(
            raw_arr, ['landcover'], {}, ignore_index=255
        )
