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

# standard imports
import dataclasses
import os
import tempfile
# third-party imports
import numpy
import pytest
# local imports
import landseg.geopipe.core as geo_core


# ----- aliases
rng = numpy.random.default_rng(42)


# ----- constants
BASE_LABEL_ARRAY = numpy.repeat([1, 2, 3, 4], 16384).reshape((1, 256, 256))

BASE_LABELSPECS: dict[str, geo_core.LabelSpecs] = {
    'base': {
        'num_cls': 4,
        'ignore_cls': [4],
        'class_name': {'1': 'WAT', '2': 'FOR', '3': 'WET', '4': 'UCL'},
    }
}

RECLASS_LABELSPECS: dict[str, geo_core.LabelSpecs] = {
    'base': {
        'num_cls': 4,
        'ignore_cls': [4],
        'class_name': {'1': 'WAT', '2': 'FOR', '3': 'WET', '4': 'UCL'},
        'reclass': {'1': [1], '2': [2, 3]},
        'reclass_name': {'1': 'WAT', '2': 'VEG'}
    }
}


# ----- `DataBlock`
def test_datablock_build_image_only_no_added_features():
    cfg = _make_config()
    inputs = _make_inputs()
    block = geo_core.DataBlock.build(inputs, cfg)

    assert block.data.image is not None
    assert block.data.valid_mask is not None
    assert block.manifest['has_label'] is False


def test_datablock_build_image_only_add_spectral():
    cfg = _make_config(add_spectral=['ndvi'])
    inputs = _make_inputs()
    block = geo_core.DataBlock.build(inputs, cfg)

    assert block.data.image.shape[0] == 8 # 7 + 1 (ndvi) = 8


def test_datablock_build_image_only_add_topo():
    cfg = _make_config(add_topo=True)
    inputs = _make_inputs(
        image_array=numpy.ones((7, 256, 256), dtype=numpy.float32),
        image_padded_dem=numpy.ones((272, 272), dtype=numpy.float32), # pad=8
    )
    block = geo_core.DataBlock.build(inputs, cfg)

    assert block.data.image.shape[0] == 11 # 7 + 4 (topo bands) = 11


def test_datablock_build_with_label_no_added_features():
    cfg = _make_config()
    inputs = _make_inputs(
        image_array=numpy.ones((7, 256, 256), dtype=numpy.float32),
        label_array=BASE_LABEL_ARRAY,
        label_specs=BASE_LABELSPECS
    )
    block = geo_core.DataBlock.build(inputs, cfg)

    assert block.manifest['has_label'] is True
    assert block.data.label is not None
    assert block.data.label_stack is not None


def test_datablock_build_full():
    cfg = _make_config(add_topo=True, add_spectral=['ndvi'])
    inputs = _make_inputs(
        image_array=numpy.ones((7, 256, 256), dtype=numpy.float32),
        image_padded_dem=numpy.ones((272, 272), dtype=numpy.float32),
        label_array=BASE_LABEL_ARRAY,
        label_specs=BASE_LABELSPECS
    )
    block = geo_core.DataBlock.build(inputs, cfg)

    assert block.data.image is not None
    assert block.data.valid_mask is not None
    assert block.data.image.shape[0] == 12 # 7 + 1 (ndvi) + 4 (topo bands) = 12
    assert block.manifest['has_label'] is True
    assert block.data.label is not None
    assert block.data.label_stack is not None

@pytest.mark.parametrize('invalid', [-999.9, numpy.nan])
def test_datablock_image_stats_handles_invalids(invalid):
    cfg = _make_config(image_nodata=invalid)

    img = numpy.ones((7, 256, 256), dtype=numpy.float32)
    invalid_idx = rng.choice(img.size, 999, replace=False)
    img.flat[invalid_idx] = invalid
    inputs = _make_inputs(image_array=img)

    block = geo_core.DataBlock.build(inputs, cfg)

    stats = block.manifest['image_stats']
    assert len(stats) == 7 # per band

    total_count = 0
    for band_idx, band in enumerate(img):
        expected_count = numpy.count_nonzero(~_is_invalid(band, invalid))
        stat = stats[f'band_{band_idx}']

        assert stat['count'] == expected_count
        assert stat['mean'] == pytest.approx(1.0)
        assert stat['m2'] == pytest.approx(0.0)

        total_count += stat['count']

    assert total_count == img.size - 999


def test_datablock_label_stack_no_reclass():
    cfg = _make_config()
    inputs = _make_inputs(
        label_array=BASE_LABEL_ARRAY,
        label_specs=BASE_LABELSPECS
    )
    block = geo_core.DataBlock.build(inputs, cfg)

    assert len(block.data.label_stack) == 1 # no reclassify
    assert block.manifest['label_num_cls'] ==  {'base': 4}
    assert block.manifest['label_ignore_cls'] == {'base': [4]}
    assert block.manifest['label_parent'] == {'base': None}
    assert block.manifest['label_parent_cls'] == {'base': None}
    assert block.manifest['label_names'] == {'base': ['WAT', 'FOR', 'WET', 'UCL']}


def test_datablocks_label_stack_w_reclass():
    cfg = _make_config()
    inputs = _make_inputs(
        label_array=BASE_LABEL_ARRAY,
        label_specs=RECLASS_LABELSPECS
    )
    block = geo_core.DataBlock.build(inputs, cfg)

    assert len(block.data.label_stack) == 4
    assert block.manifest['label_num_cls'] == {
        'base':  4,
        'base_groups': 2,
        'WAT': 1,
        'VEG': 2,
    }
    assert block.manifest['label_ignore_cls'] == {
        'base': [4],
        'base_groups': [],
        'WAT': [],
        'VEG': [],
    }
    assert block.manifest['label_parent'] == {
        'base': None,
        'base_groups': None,
        'WAT': 'base_groups',
        'VEG': 'base_groups',
    }
    assert block.manifest['label_parent_cls'] == {
        'base': None,
        'base_groups': None,
        'WAT': 1,
        'VEG': 2,
    }
    assert block.manifest['label_names'] == {
        'base': ['WAT', 'FOR', 'WET', 'UCL'],
        'base_groups': ['WAT', 'VEG'],
        'WAT': ['WAT'],
        'VEG': ['FOR', 'WET'],
    }


def test_datablock_label_stats_valid_ratio():
    cfg = _make_config()
    inputs = _make_inputs(
        label_array=BASE_LABEL_ARRAY,
        label_specs=BASE_LABELSPECS
    )
    block = geo_core.DataBlock.build(inputs, cfg)

    valid = float(numpy.mean(BASE_LABEL_ARRAY != 4)) # ignore 4 as configured
    assert block.manifest['valid_ratios'].get('base') == pytest.approx(valid)


def test_datablock_label_stats_class_count_entropy():
    cfg = _make_config()
    inputs = _make_inputs(
        label_array=BASE_LABEL_ARRAY,
        label_specs=BASE_LABELSPECS
    )
    block = geo_core.DataBlock.build(inputs, cfg)

    assert block.manifest['label_count'] == {
        'base': [16384, 16384, 16384, 0] # class 4 is ignored
    }
    assert block.manifest['label_entropy'] == {
        'base': pytest.approx(- numpy.log2(1 / 3)) # 3 valid classes each 1/3
    }


def test_datablock_save_and_load():
    cfg = _make_config()
    inputs = _make_inputs()
    block = geo_core.DataBlock.build(inputs, cfg)
    data = block.data

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, 'test_block.npz')
        block.save(fpath)

        # load block
        loaded = geo_core.DataBlock.load(fpath)
        numpy.testing.assert_array_equal(data.image, loaded.data.image)
        numpy.testing.assert_array_equal(data.valid_mask, loaded.data.valid_mask)
        assert block.manifest['block_name'] == loaded.manifest['block_name']


# ----- `DataBlockInputs`
def test_inputs_post_init_invalid_image_shape():
    with pytest.raises(ValueError, match='Image array is not of shape'):
        _make_inputs(image_array=numpy.ones((256, 256), dtype=numpy.float32))


def test_inputs_post_init_label_specs_missing():
    with pytest.raises(ValueError, match='specs not provided'):
        _make_inputs(
            label_array=BASE_LABEL_ARRAY,
            label_specs=None
        )


def test_inputs_post_init_invalid_label_shape():
    with pytest.raises(ValueError, match='Label array is not of shape'):
        _make_inputs(
            label_array=numpy.ones((256, 256)), # Not 3D
            label_specs=BASE_LABELSPECS
        )


def test_inputs_post_init_shape_mismatch():
    with pytest.raises(ValueError, match='arrays have different H / W'):
        _make_inputs(
            image_array=numpy.ones((7, 256, 256), dtype=numpy.float32),
            label_array=numpy.ones((1, 128, 128), dtype=numpy.uint8),
            label_specs=BASE_LABELSPECS
        )


def test_inputs_property_pad_dem_raise_when_not_provided():
    inputs = _make_inputs(image_padded_dem=None)
    with pytest.raises(ValueError, match='Cannot access padded DEM'):
        _ = inputs.pad_dem


def test_inputs_property_lbl_specs_raise_when_not_provided():
    inputs = _make_inputs(label_specs=None)
    with pytest.raises(ValueError, match='Cannot access label specs'):
        _ = inputs.lbl_specs


# ----- `DataBlockConfig`
def test_config_post_init_invalid_spectral_indices():
    with pytest.raises(ValueError, match='Invalid spectral indices'):
        _make_config(add_spectral=['foo'])


def test_config_post_init_missing_red_for_any_spectral():
    with pytest.raises(ValueError, match='red band missing'):
        _make_config(image_band_map={'foo': 0}, add_spectral=['ndvi'])


@pytest.mark.parametrize(
    'indice, required',
    [('ndvi', 'NIR'), ('ndmi', 'SWIR1'), ('nbr', 'SWIR2')]
)
def test_config_post_init_missing_required_bands(indice, required):
    with pytest.raises(ValueError, match=f'{required} band missing'):
        _make_config(image_band_map={'red': 0}, add_spectral=[indice])


def test_config_property_spectral_indices():
    cfg = _make_config(add_spectral=['NdVI', 'nBr'])
    assert cfg.spectral_indices == ['ndvi', 'nbr']


# ----- helpers
def _make_inputs(**overrides):
    base = geo_core.DataBlockInputs(
        block_name='test_block',
        image_array=numpy.ones((7, 256, 256), dtype=numpy.float32), # 7 bands
        image_padded_dem=None,
        label_array=None,
        label_specs=None
    )
    return dataclasses.replace(base, **overrides)


def _make_config(**overrides):
    base = geo_core.DataBlockConfig(
        image_band_map={
            'red': 0,
            'green': 1,
            'blue': 2,
            'nir': 3,
            'swir1': 4,
            'swir2': 5,
            'dem': 6
        }, # 7 bands
        image_nodata=numpy.nan,
        image_dem_pad_px=8,
        label_ignore_index=255,
        label_nodata=0,
        add_spectral=None,
        add_topo=False
    )
    return dataclasses.replace(base, **overrides)


def _is_invalid(arr, nodata):
    if isinstance(nodata, float) and numpy.isnan(nodata):
        return numpy.isnan(arr)
    return numpy.isclose(arr, nodata)
