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

'''Unit tests for block materialization and normalization utilities.'''

# standard imports
import os
# third-party imports
import numpy
import pytest
# local imports
import landseg.geopipe.contracts as contracts
import landseg.geopipe.prepare.data_context as data_context
import landseg.geopipe.prepare.materialize_blocks.materialize as mat


# ----- `_normalize_image` tests
def test_normalize_image_math():
    '''
    Given: Raw image array, valid pixel mask, and global band stats.
    When: Running _normalize_image.
    Then: Correctly apply z-score normalization and replace invalid
        pixels with global mean.
    '''
    raw_img = numpy.array(
        [[[10.0, 20.0], [30.0, 40.0]]], dtype=numpy.float32
    )
    # pixel (1, 1) is invalid
    mask = numpy.array([[True, True], [True, False]], dtype=bool)

    stats: contracts.ImageBandStats = {
        'total_count': 100,
        'current_mean': 20.0,
        'accum_m2': 100.0,
        'std': 5.0,
    }
    global_stats = {'band_0': stats}

    norm_img = mat._normalize_image(raw_img, mask, global_stats)

    # pixel 0,0: (10.0 - 20.0) / 5.0 = -2.0
    # pixel 0,1: (20.0 - 20.0) / 5.0 = 0.0
    # pixel 1,0: (30.0 - 20.0) / 5.0 = 2.0
    # pixel 1,1: invalid, replaced by mean (20.0), so 0.0
    expected = numpy.array(
        [[[-2.0, 0.0], [2.0, 0.0]]], dtype=numpy.float32
    )
    assert numpy.allclose(norm_img, expected)


# ----- `_reclassify_labels` tests
def test_reclassify_labels_multi_head():
    '''
    Given: Raw label array and resolved reclassification groups.
    When: Running _reclassify_labels.
    Then: Construct base layer, grouping layer, and child slice layers.
    '''
    # raw label array with 1-based pixel values:
    # 1: veg_conifer, 2: veg_decid, 3: water, 255: ignore
    raw_arr = numpy.array([
        [1, 2],
        [3, 255],
    ], dtype=numpy.uint8)

    # 0-based group ids and 0-based source classes:
    # group 0 (VEG): classes (0, 1) -> pixel values (1, 2)
    # group 1 (WAT): class (2,) -> pixel value (3,)
    target_reclass = {
        'landcover': {0: (0, 1), 1: (2,)}
    }

    stack = mat._reclassify_labels(
        raw_labels=[raw_arr],
        label_layer_names=['landcover'],
        target_reclass=target_reclass,
        ignore_index=255,
    )
    # expected 4 layers:
    # 0: base layer (1..3)
    # 1: grouping layer (group 0 for 1,2; group 1 for 3)
    # 2: child slice 0 (classes 0 and 1)
    # 3: child slice 1 (class 2)
    assert stack.shape == (4, 2, 2)

    # base layer
    assert numpy.array_equal(stack[0], raw_arr)

    # grouping layer
    assert stack[1, 0, 0] == 0
    assert stack[1, 0, 1] == 0
    assert stack[1, 1, 0] == 1
    assert stack[1, 1, 1] == 255

    # child slice 0 (VEG: 1 -> 0, 2 -> 1, others -> 255)
    assert stack[2, 0, 0] == 0
    assert stack[2, 0, 1] == 1
    assert stack[2, 1, 0] == 255
    assert stack[2, 1, 1] == 255

    # child slice 1 (WAT: 3 -> 0, others -> 255)
    assert stack[3, 0, 0] == 255
    assert stack[3, 0, 1] == 255
    assert stack[3, 1, 0] == 0
    assert stack[3, 1, 1] == 255


def test_reclassify_labels_pass_through():
    '''
    Given: 3D label array without target reclassification.
    When: Running _reclassify_labels.
    Then: Return unchanged layers in the stack.
    '''
    raw_arr = numpy.zeros((1, 4, 4), dtype=numpy.uint8)
    stack = mat._reclassify_labels(
        raw_labels=raw_arr,
        label_layer_names=['landcover'],
        target_reclass={},
        ignore_index=255,
    )
    assert stack.shape == (1, 4, 4)
    assert numpy.array_equal(stack, raw_arr)


def test_reclassify_labels_invalid_dim():
    '''
    Given: 1D label array with unsupported dimensionality.
    When: Running _reclassify_labels.
    Then: Raise ValueError.
    '''
    raw_arr = numpy.zeros((10,), dtype=numpy.uint8)
    with pytest.raises(ValueError, match='Expected 2D or 3D label array'):
        mat._reclassify_labels(
            raw_labels=raw_arr,
            label_layer_names=['landcover'],
            target_reclass={},
            ignore_index=255,
        )


# ----- `_purge` tests
def test_purge_removes_stale_files(tmp_path):
    '''
    Given: Target directory containing expected and stale files.
    When: Running _purge.
    Then: Remove stale files and preserve expected ones.
    '''
    target_dir = tmp_path / 'normalized_blocks'
    os.makedirs(target_dir, exist_ok=True)

    expected_file = target_dir / 'block_1.npz'
    stale_file = target_dir / 'block_stale.npz'

    expected_file.touch()
    stale_file.touch()

    filenames_to_keep = ['block_1.npz']
    purged_count = mat._purge(filenames_to_keep, str(target_dir))

    assert purged_count == 1
    assert os.path.exists(expected_file)
    assert not os.path.exists(stale_file)


# ----- `materialize_blocks` tests
def test_materialize_blocks_orchestration(tmp_path, mocker):
    '''
    Given: Input block paths, global stats, and dataset context.
    When: Running materialize_blocks.
    Then: Execute block materialization and return indexed output paths.
    '''
    out_dir = str(tmp_path / 'materialized')
    input_blocks = {'path/to/block_0_0.npz'}

    def fake_run(jobs):
        for _func, (b, _stats, out_d, _ctx), _kwargs in jobs:
            (tmp_path / 'materialized' / os.path.basename(b)).touch()

    mock_run = mocker.patch(
        'landseg.utils.ParallelExecutor.run', side_effect=fake_run
    )

    os.makedirs(out_dir, exist_ok=True)

    mock_ctx = mocker.Mock(spec=data_context.DatasetContext)

    indexed, purged = mat.materialize_blocks(
        input_blocks=input_blocks,
        stats={},
        context=mock_ctx,
        output_dir=out_dir,
    )

    assert 'block_0_0' in indexed
    assert purged == 0
    assert mock_run.called
