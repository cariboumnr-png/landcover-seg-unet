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

'''Unit tests for core DataBlock, DataBlockArrays, and DataBlockManifest.'''

# standard imports
import os
import tempfile
# third-party imports
import numpy
import pytest
# local imports
import landseg.geopipe.core as geo_core


# ----- `DataBlockArrays` tests
def test_datablock_arrays_validate_success():
    '''
    Given: Valid image, label, and mask arrays.
    When: Running validate.
    Then: Validation succeeds without error.
    '''
    arrays = geo_core.DataBlockArrays(
        image=numpy.ones((4, 64, 64), dtype=numpy.float32),
        label=numpy.ones((1, 64, 64), dtype=numpy.uint8),
        valid_mask=numpy.ones((64, 64), dtype=bool),
    )
    arrays.validate()


def test_datablock_arrays_validate_invalid_image():
    '''
    Given: An image array of wrong dimensionality.
    When: Running validate.
    Then: Raise ValueError.
    '''
    arrays = geo_core.DataBlockArrays(
        image=numpy.ones((64, 64), dtype=numpy.float32),
        label=numpy.ones((1, 64, 64), dtype=numpy.uint8),
        valid_mask=numpy.ones((64, 64), dtype=bool),
    )
    with pytest.raises(ValueError, match='Image array is not of shape'):
        arrays.validate()


def test_datablock_arrays_validate_invalid_mask():
    '''
    Given: A mask array of wrong dimensionality.
    When: Running validate.
    Then: Raise ValueError.
    '''
    arrays = geo_core.DataBlockArrays(
        image=numpy.ones((4, 64, 64), dtype=numpy.float32),
        label=numpy.ones((1, 64, 64), dtype=numpy.uint8),
        valid_mask=numpy.ones((1, 64, 64), dtype=bool),
    )
    with pytest.raises(ValueError, match='Valid mask array is not of shape'):
        arrays.validate()


def test_datablock_arrays_validate_shape_mismatch():
    '''
    Given: Image and mask arrays with mismatched dimensions.
    When: Running validate.
    Then: Raise ValueError.
    '''
    arrays = geo_core.DataBlockArrays(
        image=numpy.ones((4, 64, 64), dtype=numpy.float32),
        label=numpy.ones((1, 64, 64), dtype=numpy.uint8),
        valid_mask=numpy.ones((32, 32), dtype=bool),
    )
    with pytest.raises(ValueError, match='mismatched H / W'):
        arrays.validate()


# ----- `DataBlock` tests
def test_datablock_default_initialization():
    '''
    Given: No constructor arguments.
    When: Instantiating DataBlock.
    Then: Correctly initialize empty arrays and default manifest.
    '''
    block = geo_core.DataBlock()
    assert block.data is not None
    assert block.manifest['has_label'] is False
    assert block.manifest['block_name'] == ''


def test_datablock_empty_manifest():
    '''
    Given: A block name.
    When: Calling empty_manifest.
    Then: Return a valid DataBlockManifest with defaults.
    '''
    manifest = geo_core.DataBlock.empty_manifest('block_01')
    assert manifest['block_name'] == 'block_01'
    assert manifest['label_ignore_index'] == 255
    assert manifest['has_label'] is False


def test_datablock_save_and_load():
    '''
    Given: A DataBlock with arrays and manifest.
    When: Saving to and loading back from disk as an npz payload.
    Then: Perfectly preserve the arrays and manifest metadata.
    '''
    arrays = geo_core.DataBlockArrays(
        image=numpy.ones((4, 32, 32), dtype=numpy.float32),
        label=numpy.ones((1, 32, 32), dtype=numpy.uint8),
        valid_mask=numpy.ones((32, 32), dtype=bool),
    )
    manifest = geo_core.DataBlock.empty_manifest('test_block')
    manifest['has_label'] = True
    manifest['image_stats'] = {'band_0': {'count': 1024, 'mean': 1.0, 'm2': 0.0}}

    block = geo_core.DataBlock(data=arrays, manifest=manifest)

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, 'test_block.npz')
        block.save(fpath)

        loaded = geo_core.DataBlock.load(fpath)
        numpy.testing.assert_array_equal(block.data.image, loaded.data.image)
        numpy.testing.assert_array_equal(block.data.label, loaded.data.label)
        numpy.testing.assert_array_equal(
            block.data.valid_mask, loaded.data.valid_mask
        )
        assert loaded.manifest['block_name'] == 'test_block'
        assert loaded.manifest['has_label'] is True
        assert loaded.manifest['image_stats']['band_0']['count'] == 1024


def test_datablock_save_invalid_extension():
    '''
    Given: A file path without .npz extension.
    When: Saving a DataBlock.
    Then: Raise ValueError.
    '''
    block = geo_core.DataBlock()
    with pytest.raises(ValueError, match='Output path must end with .npz'):
        block.save('invalid_path.bin')
