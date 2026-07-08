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
# pylint: disable=protected-access

'''Unit tests for DataBlock, DataBlockInputs, and DataBlockConfig.'''

import tempfile
import os
import numpy
import pytest
import landseg.geopipe.core.foundation_data_block as fdb

def test_context_validation_success():
    cfg = get_valid_config()
    cfg = fdb.DataBlockConfig(
        image_band_map=cfg.image_band_map,
        image_nodata=cfg.image_nodata,
        image_dem_pad_px=cfg.image_dem_pad_px,
        label_ignore_index=cfg.label_ignore_index,
        label_nodata=cfg.label_nodata,
        add_spectral=['ndvi', 'ndmi', 'nbr'],
        add_topo=True
    )
    img = numpy.ones((7, 256, 256), dtype=numpy.float32)
    dem = numpy.ones((256 + 16, 256 + 16), dtype=numpy.float32)

    _ = fdb.DataBlockInputs(
        block_name='test_block',
        image_array=img,
        image_padded_dem=dem,
        label_array=None,
        label_specs=None
    )
    assert cfg.add_topo is True
    assert cfg.spectral_indices == ['ndvi', 'ndmi', 'nbr']

def test_context_validation_invalid_image_shape():
    img = numpy.ones((256, 256), dtype=numpy.float32) # Not 3D

    with pytest.raises(ValueError, match='Image array is not of shape'):
        fdb.DataBlockInputs(
            block_name='test_block',
            image_array=img,
            image_padded_dem=None,
            label_array=None,
            label_specs=None
        )

def test_context_validation_invalid_label_shape():
    img = numpy.ones((7, 256, 256), dtype=numpy.float32)
    lbl = numpy.ones((256, 256), dtype=numpy.uint8) # Not 3D

    with pytest.raises(ValueError, match='Label array is not of shape'):
        fdb.DataBlockInputs(
            block_name='test_block',
            image_array=img,
            image_padded_dem=None,
            label_array=lbl,
            label_specs=None
        )

def test_context_validation_shape_mismatch():
    img = numpy.ones((7, 256, 256), dtype=numpy.float32)
    lbl = numpy.ones((1, 128, 128), dtype=numpy.uint8) # Mismatch

    with pytest.raises(ValueError, match='Image and label arrays have different'):
        fdb.DataBlockInputs(
            block_name='test_block',
            image_array=img,
            image_padded_dem=None,
            label_array=lbl,
            label_specs=None
        )

def test_context_validation_invalid_spectral_index():
    cfg = get_valid_config()
    with pytest.raises(ValueError, match='Invalid spectral indices'):
        fdb.DataBlockConfig(
            image_band_map=cfg.image_band_map,
            image_nodata=cfg.image_nodata,
            image_dem_pad_px=cfg.image_dem_pad_px,
            label_ignore_index=cfg.label_ignore_index,
            label_nodata=cfg.label_nodata,
            add_spectral=['invalid_index'],
            add_topo=False
        )

def test_context_validation_missing_red_for_spectral():
    cfg = get_valid_config(image_band_map={'green': 0, 'blue': 1}) # No red
    with pytest.raises(ValueError, match='red band missing'):
        fdb.DataBlockConfig(
            image_band_map=cfg.image_band_map,
            image_nodata=cfg.image_nodata,
            image_dem_pad_px=cfg.image_dem_pad_px,
            label_ignore_index=cfg.label_ignore_index,
            label_nodata=cfg.label_nodata,
            add_spectral=['ndvi'],
            add_topo=False
        )

def test_context_validation_missing_nir_for_ndvi():
    cfg = get_valid_config(image_band_map={'red': 0, 'green': 1}) # No nir
    with pytest.raises(ValueError, match='NIR band missing'):
        fdb.DataBlockConfig(
            image_band_map=cfg.image_band_map,
            image_nodata=cfg.image_nodata,
            image_dem_pad_px=cfg.image_dem_pad_px,
            label_ignore_index=cfg.label_ignore_index,
            label_nodata=cfg.label_nodata,
            add_spectral=['ndvi'],
            add_topo=False
        )

def test_datablock_build_unlabeled_success():
    cfg = get_valid_config()
    cfg = fdb.DataBlockConfig(
        image_band_map=cfg.image_band_map,
        image_nodata=cfg.image_nodata,
        image_dem_pad_px=cfg.image_dem_pad_px,
        label_ignore_index=cfg.label_ignore_index,
        label_nodata=cfg.label_nodata,
        add_spectral=['ndvi'],
        add_topo=True
    )
    img = numpy.ones((7, 256, 256), dtype=numpy.float32)
    dem = numpy.ones((256 + 16, 256 + 16), dtype=numpy.float32)

    inputs = fdb.DataBlockInputs(
        block_name='test_block',
        image_array=img,
        image_padded_dem=dem,
        label_array=None,
        label_specs=None
    )

    block = fdb.DataBlock.build(inputs, cfg)
    assert block.data.image is not None
    assert block.data.valid_mask is not None
    assert block.manifest['has_label'] is False
    # Verify spectral band added (original 7 bands + 1 spectral + 4 topo = 12 bands)
    assert block.data.image.shape[0] == 12

def test_datablock_build_labeled_success():
    cfg = get_valid_config()
    img = numpy.ones((7, 256, 256), dtype=numpy.float32)
    lbl = numpy.ones((1, 256, 256), dtype=numpy.uint8)

    label_specs = {
        'head_1': {
            'num_cls': 2,
            'ignore_cls': [255],
            'reclass': {'1': [1]}
        }
    }

    inputs = fdb.DataBlockInputs(
        block_name='test_block',
        image_array=img,
        image_padded_dem=None,
        label_array=lbl,
        label_specs=label_specs
    )

    block = fdb.DataBlock.build(inputs, cfg)
    assert block.manifest['has_label'] is True
    assert block.data.label is not None
    assert block.data.label_stack is not None

def test_datablock_save_and_load():
    cfg = get_valid_config()
    img = numpy.ones((7, 256, 256), dtype=numpy.float32)

    inputs = fdb.DataBlockInputs(
        block_name='test_block',
        image_array=img,
        image_padded_dem=None,
        label_array=None,
        label_specs=None
    )
    block = fdb.DataBlock.build(inputs, cfg)
    data = block.data

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, 'test_block.npz')
        block.save(fpath)

        # Load block
        loaded = fdb.DataBlock.load(fpath)
        loaded_data = loaded.data
        numpy.testing.assert_array_equal(data.image, loaded_data.image)
        numpy.testing.assert_array_equal(data.valid_mask, loaded_data.valid_mask)
        assert block.manifest['block_name'] == loaded.manifest['block_name']

# Helper to build a valid config
def get_valid_config(image_band_map=None):
    if image_band_map is None:
        image_band_map = {
            'red': 0,
            'green': 1,
            'blue': 2,
            'nir': 3,
            'swir1': 4,
            'swir2': 5,
            'dem': 6
        }
    return fdb.DataBlockConfig(
        image_band_map=image_band_map,
        image_nodata=numpy.nan,
        image_dem_pad_px=8,
        label_ignore_index=255,
        label_nodata=0,
        add_spectral=None,
        add_topo=False
    )
