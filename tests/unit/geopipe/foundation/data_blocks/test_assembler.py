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

'''Unit tests for assembler modules build blocks API.'''

# standard imports
import json
import os
# third-party imports
import numpy
import pytest
import rasterio
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.common.alias as alias
import landseg.geopipe.foundation.data_blocks.assembler as assembler
import landseg.geopipe.utils as geo_utils


# ----- fixtures
@pytest.fixture(name='assembler_config_json')
def fixture_assembler_config_json(tmp_path):
    '''Mock metadata source config file JSON.'''
    data = {
        'image_band_map': {
            'red': 0,
            'green': 1,
            'blue': 2,
            'nir': 3,
            'dem': 4
        },
        'label_specs': {
            'class_head': {
                'num_cls': 2,
                'ignore_cls': [255]
            }
        },
        'label_color_map': {
            '1': [0, 255, 0]
        }
    }
    fpath = tmp_path / 'config.json'
    with open(fpath, 'w', encoding='UTF-8') as f:
        json.dump(data, f)
    return fpath


# ----- integrity checks
def test_check_npz_integrity_success(tmp_path):
    '''
    Given: A valid DataBlock saved as an npz file on disk.
    When: Running check_npz_integrity.
    Then: Return a dictionary flagging the file as valid.
    '''
    fpath = tmp_path / 'test.npz'
    img = numpy.ones((5, 8, 8), dtype=numpy.float32)
    cfg = geo_core.DataBlockConfig(
        image_band_map={
            'red': 0,
            'green': 1,
            'blue': 2,
            'nir': 3,
            'dem': 4,
        },
        image_nodata=numpy.nan,
        image_dem_pad_px=8,
        label_ignore_index=255
    )
    inputs = geo_core.DataBlockInputs(
        block_name='test_block',
        image_array=img,
        image_padded_dem=None,
        label_array=None,
        label_specs=None
    )
    block = geo_core.DataBlock.build(inputs, cfg)
    block.save(str(fpath))
    res = assembler.check_npz_integrity((0, 0), str(fpath))
    assert res == {(0, 0): True}


def test_check_npz_integrity_missing():
    '''
    Given: A non-existent file path.
    When: Running check_npz_integrity.
    Then: Return a dictionary flagging the file as invalid.
    '''
    res = assembler.check_npz_integrity((0, 0), 'non_existent_file.npz')
    assert res == {(0, 0): False}


def test_check_npz_integrity_corrupted(tmp_path):
    '''
    Given: A corrupted text file acting as a mock npz file.
    When: Running check_npz_integrity.
    Then: Return a dictionary flagging the file as invalid.
    '''
    fpath = tmp_path / 'corrupt.npz'
    with open(fpath, 'w', encoding='UTF-8') as f:
        f.write('not a zip file')
    res = assembler.check_npz_integrity((0, 0), str(fpath))
    assert res == {(0, 0): False}


# ----- single block construction
def test_build_single_block_success(dummy_geotiff_factory):
    '''
    Given: Valid image and label rasters with DEM and spectral config.
    When: Running build_single_block.
    Then: Return a DataBlock containing calculated indices and labels.
    '''
    img_path = str(dummy_geotiff_factory('image.tif', 16, 16, 5))
    lbl_path = str(dummy_geotiff_factory('label.tif', 16, 16, 1))

    window = alias.RasterWindow(col_off=4, row_off=4, width=8, height=8) # type: ignore

    label_specs = {
        'class_head': {
            'num_cls': 2,
            'ignore_cls': [255]
        }
    }

    inputs = assembler.RasterReadInput(
        image_fpath=img_path,
        image_window=window,
        image_band_map={
            'red': 0,
            'green': 1,
            'blue': 2,
            'nir': 3,
            'dem': 4,
        },
        image_dem_pad_px=2,
        label_fpath=lbl_path,
        label_window=window,
        label_specs=label_specs # type: ignore
    )

    block = assembler.build_single_block(
        name='block_4_4',
        inputs=inputs,
        ignore_index=255,
        add_spectral=['ndvi'],
        add_topo=True
    )
    assert block.manifest['block_name'] == 'block_4_4'
    assert block.manifest['has_label'] is True
    assert block.data.image.shape == (5 + 1 + 4, 8, 8)


def test_build_single_block_defaults(dummy_geotiff_factory):
    '''
    Given: Valid image and label rasters without extra features requested.
    When: Running build_single_block with default options.
    Then: Return a block containing only the original image bands.
    '''
    img_path = str(dummy_geotiff_factory('image2.tif', 16, 16, 5))
    lbl_path = str(dummy_geotiff_factory('label2.tif', 16, 16, 1))

    window = alias.RasterWindow(col_off=4, row_off=4, width=8, height=8) # type: ignore
    label_specs = {
        'class_head': {
            'num_cls': 2,
            'ignore_cls': [255]
        }
    }

    inputs = assembler.RasterReadInput(
        image_fpath=img_path,
        image_window=window,
        image_band_map={
            'red': 0,
            'green': 1,
            'blue': 2,
            'nir': 3,
            'dem': 4,
        },
        image_dem_pad_px=2,
        label_fpath=lbl_path,
        label_window=window,
        label_specs=label_specs # type: ignore
    )

    block = assembler.build_single_block(
        name='block_defaults',
        inputs=inputs
    )
    assert block.data.image.shape == (5, 8, 8)


# ----- batch block construction
def test_build_blocks_orchestrator(
    dummy_geotiff_factory,
    assembler_config_json,
    tmp_path
):
    '''
    Given: Configuration file paths and image/label window maps.
    When: Running build_blocks orchestrator.
    Then: Save built DataBlocks to the output path.
    '''
    img_path = str(dummy_geotiff_factory('image.tif', 16, 16, 5))
    lbl_path = str(dummy_geotiff_factory('label.tif', 16, 16, 1))

    inputs = assembler.BlockBuildingInput(
        output_root=str(tmp_path / 'blocks'),
        image_fpath=img_path,
        label_fpath=lbl_path,
        config_fpath=str(assembler_config_json)
    )

    image_windows = {
        (0, 0): alias.RasterWindow(col_off=0, row_off=0, width=8, height=8), # type: ignore
        (0, 8): alias.RasterWindow(col_off=8, row_off=0, width=8, height=8), # type: ignore
        (8, 0): alias.RasterWindow(col_off=0, row_off=8, width=8, height=8), # type: ignore
        (8, 8): alias.RasterWindow(col_off=8, row_off=8, width=8, height=8), # type: ignore
    }

    label_windows = dict(image_windows)
    context = assembler.BlockBuildingContext(
        image=image_windows,
        label=label_windows
    )

    label_specs = {
        'class_head': {
            'num_cls': 2,
            'ignore_cls': [255]
        }
    }

    config = assembler.BlockBuildingConfig(
        ignore_index=255,
        dem_pad_px=2,
        block_size=(8, 8),
        image_band_map={
            'red': 0,
            'green': 1,
            'blue': 2,
            'nir': 3,
            'dem': 4,
        },
        label_specs=label_specs, # type: ignore
        add_spectral=['ndvi'],
        add_topo=True
    )

    result = assembler.build_blocks(
        inputs=inputs,
        context=context,
        config=config
    )

    assert len(result.coords_created) == 4
    assert result.stats['blocks_created'] == 4
    assert result.stats['blocks_on_disk_before'] == 0
    assert result.label_color_map == {'1': [0, 255, 0]}

    for coord in image_windows:
        name = geo_utils.xy_name(coord)
        assert os.path.exists(os.path.join(inputs.output_root, f'{name}.npz'))


# ----- test block construction
def test_build_test_block_success(dummy_geotiff_factory, tmp_path):
    '''
    Given: Large label and image inputs meeting class coverage requirements.
    When: Running build_test_block.
    Then: Return the path to a serialized DataBlock with valid coverage.
    '''
    img_path = str(dummy_geotiff_factory('image.tif', 16, 16, 5))
    lbl_path = str(dummy_geotiff_factory('label.tif', 16, 16, 1))

    with rasterio.open(lbl_path, 'r+') as src:
        arr = numpy.ones((1, 16, 16), dtype=numpy.uint8)
        arr[0, 0:8, :] = 1
        arr[0, 8:16, :] = 2
        src.write(arr)

    window = alias.RasterWindow(col_off=0, row_off=0, width=16, height=16) # type: ignore

    label_specs = {
        'class_head': {
            'num_cls': 2,
            'ignore_cls': [255]
        }
    }

    read_input = assembler.RasterReadInput(
        image_fpath=img_path,
        image_window=window,
        image_band_map={
            'red': 0,
            'green': 1,
            'blue': 2,
            'nir': 3,
            'dem': 4,
        },
        image_dem_pad_px=2,
        label_fpath=lbl_path,
        label_window=window,
        label_specs=label_specs # type: ignore
    )

    inputs = {'test_block': read_input}

    fpath = assembler.build_test_block(
        save_dpath=str(tmp_path / 'test_blocks'),
        inputs=inputs,
        target_head='class_head',
        valid_px_per=0.5,
        need_all_classes=True
    )

    assert fpath is not None
    assert os.path.exists(fpath)
