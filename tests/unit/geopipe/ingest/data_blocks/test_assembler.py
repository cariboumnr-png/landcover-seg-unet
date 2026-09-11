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

'''Unit tests for assembler modules build blocks API.'''

# standard imports
import json
import os
# third-party imports
import numpy
import pytest
import rasterio
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.ingest.common.alias as alias
import landseg.geopipe.ingest.data_blocks.assembler as assembler
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
    img_path = str(dummy_geotiff_factory(
        filename='image.tif', width=16, height=16, bands=5
    ))
    lbl_path = str(dummy_geotiff_factory(
        filename='label.tif', width=16, height=16, bands=1
    ))

    window = alias.RasterWindow(4, 4, 8, 8)  # type: ignore

    label_specs: dict[str, geo_core.CategoricalSpecs] = {
        'class_head': {
            'num_cls': 2,
            'ignore_cls': [255],
            'index_base': 0,
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
        label_specs=label_specs
    )

    block = assembler.build_single_block(
        name='block_4_4',
        inputs=inputs,
        ignore_index=255,
        add_spectral=['ndvi'],
        add_topo=['slope', 'aspect', 'tpi']
    )
    assert block.manifest['block_name'] == 'block_4_4'
    assert block.manifest['has_label'] is True
    assert block.data.image.shape == (5 + 1 + 4, 8, 8)


def test_build_single_block_defaults(dummy_geotiff_factory):
    '''
    Given: Valid rasters without extra features requested.
    When: Running build_single_block with default options.
    Then: Return a block containing only the original image bands.
    '''
    img_path = str(dummy_geotiff_factory(
        filename='image2.tif', width=16, height=16, bands=5
    ))
    lbl_path = str(dummy_geotiff_factory(
        filename='label2.tif', width=16, height=16, bands=1
    ))

    window = alias.RasterWindow(4, 4, 8, 8)  # type: ignore
    label_specs: dict[str, geo_core.CategoricalSpecs] = {
        'class_head': {
            'num_cls': 2,
            'ignore_cls': [255],
            'index_base': 0,
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
        label_specs=label_specs
    )

    block = assembler.build_single_block(
        name='block_4_4_def',
        inputs=inputs,
        ignore_index=255
    )
    assert block.manifest['block_name'] == 'block_4_4_def'
    assert block.manifest['has_label'] is True
    assert block.data.image.shape == (5, 8, 8)


# ----- batch block construction
def test_build_blocks_orchestrator(
    dummy_geotiff_factory,
    tmp_path
):
    '''
    Given: Configuration file paths and image/label window maps.
    When: Running build_blocks orchestrator.
    Then: Save built DataBlocks to the output path.
    '''
    img_path = str(dummy_geotiff_factory(
        filename='image.tif', width=16, height=16, bands=5
    ))
    lbl_path = str(dummy_geotiff_factory(
        filename='label.tif', width=16, height=16, bands=1
    ))

    inputs = assembler.BlockBuildingInput(
        output_root=str(tmp_path / 'blocks'),
        image_fpath=img_path,
        label_fpath=lbl_path,
    )

    image_windows = {
        (0, 0): alias.RasterWindow(0, 0, 8, 8),  # type: ignore
        (0, 8): alias.RasterWindow(8, 0, 8, 8),  # type: ignore
        (8, 0): alias.RasterWindow(0, 8, 8, 8),  # type: ignore
        (8, 8): alias.RasterWindow(8, 8, 8, 8),  # type: ignore
    }

    label_windows = dict(image_windows)
    context = assembler.BlockBuildingContext(
        image=image_windows,
        label=label_windows
    )

    label_specs: dict[str, geo_core.CategoricalSpecs] = {
        'class_head': {
            'num_cls': 2,
            'ignore_cls': [255],
            'index_base': 0,
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
        label_specs=label_specs,
        add_spectral=['ndvi'],
        add_topo=['slope', 'aspect', 'tpi']
    )

    result = assembler.build_blocks(
        inputs=inputs,
        context=context,
        config=config,
        policy=artifacts.LifecyclePolicy.REBUILD,
    )

    assert len(result.coords_created) == 4
    assert result.stats['blocks_created'] == 4
    assert result.stats['blocks_on_disk_before'] == 0
    assert result.label_color_map is None

    for coord in image_windows:
        name = geo_utils.xy_name(coord)
        assert os.path.exists(os.path.join(inputs.output_root, f'{name}.npz'))


# ----- test block construction
def test_build_test_block_success(dummy_geotiff_factory, tmp_path):
    '''
    Given: Large label and image inputs meeting coverage requirements.
    When: Running raster_assembler.
    Then: Return path to serialized DataBlock with valid coverage.
    '''
    img_path = str(dummy_geotiff_factory(
        filename='image.tif', width=16, height=16, bands=5
    ))
    lbl_path = str(dummy_geotiff_factory(
        filename='label.tif', width=16, height=16, bands=1
    ))

    with rasterio.open(lbl_path, 'r+') as src:
        arr = numpy.ones((1, 16, 16), dtype=numpy.uint8)
        arr[0, 0:8, :] = 1
        arr[0, 8:16, :] = 2
        src.write(arr)

    window = alias.RasterWindow(0, 0, 16, 16)  # type: ignore

    label_specs: dict[str, geo_core.CategoricalSpecs] = {
        'class_head': {
            'num_cls': 2,
            'ignore_cls': [255],
            'index_base': 1,
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
        label_specs=label_specs,
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


# ----- `read_label_specs` and `read_schemes` tests
def test_read_label_specs_dataset_level_tags(dummy_geotiff_factory):
    '''
    Given: A single-band label raster with dataset-level metadata tags.
    When: Reading label specs via assembler.
    Then: Return parsed label specifications dictionary.
    '''
    lbl_path = str(dummy_geotiff_factory(
        filename='single_lbl.tif', width=16, height=16, bands=1
    ))
    with rasterio.open(lbl_path, 'r+') as src:
        src.set_band_description(1, 'landcover')
        src.update_tags(
            num_cls='2',
            ignore_cls='[255]',
            index_base='1',
            class_name="{'1': 'forest', '2': 'water'}",
        )

    specs = assembler.read_label_specs(lbl_path)
    assert 'landcover' in specs
    assert specs['landcover']['num_cls'] == 2
    assert specs['landcover']['ignore_cls'] == [255]
    assert specs['landcover']['index_base'] == 1
    assert specs['landcover'].get('class_name') == {
        '1': 'forest',
        '2': 'water',
    }


def test_read_schemes_nested_and_flat(dummy_geotiff_factory):
    '''
    Given: A raster with dataset-level and band-level schemes tags.
    When: Reading schemes via assembler.
    Then: Correctly parse and merge schemes dictionaries.
    '''
    feat_path = str(dummy_geotiff_factory(
        filename='feat_schemes.tif', width=16, height=16, bands=1
    ))
    with rasterio.open(feat_path, 'r+') as src:
        src.update_tags(
            schemes="{'s2': {'rgb': ['blue', 'green', 'red']}}"
        )

    schemes = assembler.read_schemes(feat_path)
    assert 's2' in schemes
    assert schemes['s2']['rgb'] == ['blue', 'green', 'red']
