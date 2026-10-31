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

'''Unit tests for RasterReadOutput and build_data_block.'''

# standard imports
import dataclasses
import pathlib
# third-party imports
import numpy
import pytest
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.ingest.blocks.assembler.builder as builder
import landseg.geopipe.ingest.blocks.assembler.io as io


# ----- aliases
rng = numpy.random.default_rng(42)


# ----- constants
BASE_LABEL_ARRAY = numpy.repeat([1, 2, 3, 4], 16384).reshape((1, 256, 256))

BASE_LABELSPECS: dict[str, geo_core.CategoricalSpec] = {
    'base': {
        'num_cls': 4,
        'ignore_cls': (4,),
        'class_name': {'1': 'WAT', '2': 'FOR', '3': 'WET', '4': 'UCL'},
        'index_base': 1,
    }
}


# ----- `build_data_block` tests
def test_build_data_block_image_only_no_added_features():
    '''
    Given: An image-only input block.
    When: Building a DataBlock without any added features.
    Then: Return a block containing valid image arrays and no label.
    '''
    raster_data = _make_raster_data()
    block = builder.build_data_block('test_block', raster_data)

    assert block.data.image is not None
    assert block.data.valid_mask is not None
    assert block.manifest['has_label'] is False


def test_build_data_block_image_only_add_spectral():
    '''
    Given: An image-only input block.
    When: Building a DataBlock requesting ndvi spectral index.
    Then: Append ndvi as an extra band in the image data.
    '''
    raster_data = _make_raster_data()
    block = builder.build_data_block(
        'test_block', raster_data, add_spectral=['ndvi']
    )

    assert block.data.image.shape[0] == 8


def test_build_data_block_image_only_add_topo():
    '''
    Given: An image input block with padded DEM elevation data.
    When: Building a DataBlock requesting topographic indices.
    Then: Append slope, aspect cos, aspect sin, and tpi bands.
    '''
    raster_data = _make_raster_data(
        image_array=numpy.ones((7, 256, 256), dtype=numpy.float32),
        image_padded_dem=numpy.ones((272, 272), dtype=numpy.float32),
    )
    block = builder.build_data_block(
        'test_block', raster_data, add_topo=['slope', 'aspect', 'tpi']
    )

    assert block.data.image.shape[0] == 11


def test_build_data_block_with_label_no_added_features():
    '''
    Given: Input block containing both image and label arrays.
    When: Building a DataBlock.
    Then: Return a block containing the parsed multi-channel label stack.
    '''
    raster_data = _make_raster_data(
        image_array=numpy.ones((7, 256, 256), dtype=numpy.float32),
        label_array=BASE_LABEL_ARRAY,
        label_specs=BASE_LABELSPECS,
    )
    block = builder.build_data_block('test_block', raster_data)

    assert block.manifest['has_label'] is True
    assert block.data.label is not None


def test_build_data_block_full():
    '''
    Given: Input block with padded DEM, label, and spectral config.
    When: Building a DataBlock.
    Then: Generate image indices, DEM features, and label stacks.
    '''
    raster_data = _make_raster_data(
        image_array=numpy.ones((7, 256, 256), dtype=numpy.float32),
        image_padded_dem=numpy.ones((272, 272), dtype=numpy.float32),
        label_array=BASE_LABEL_ARRAY,
        label_specs=BASE_LABELSPECS,
    )
    block = builder.build_data_block(
        'test_block',
        raster_data,
        add_spectral=['ndvi'],
        add_topo=['slope', 'aspect', 'tpi'],
    )

    assert block.data.image is not None
    assert block.data.valid_mask is not None
    assert block.data.image.shape[0] == 12
    assert block.manifest['has_label'] is True
    assert block.data.label is not None


@pytest.mark.parametrize('invalid', [-999.9, numpy.nan])
def test_build_data_block_image_stats_handles_invalids(invalid):
    '''
    Given: An image containing invalid values.
    When: Building a DataBlock.
    Then: Ignore the invalid values during band mean/std calculations.
    '''
    img = numpy.ones((7, 256, 256), dtype=numpy.float32)
    invalid_idx = rng.choice(img.size, 999, replace=False)
    img.flat[invalid_idx] = invalid
    raster_data = _make_raster_data(
        image_array=img,
        image_nodata=invalid,
    )
    block = builder.build_data_block('test_block', raster_data)

    stats = block.manifest['image_stats']
    assert len(stats) == 7

    total_count = 0
    for band_idx, band in enumerate(img):
        expected_count = numpy.count_nonzero(~_is_invalid(band, invalid))
        stat = stats[f'band_{band_idx}']

        assert stat['count'] == expected_count
        assert stat['mean'] == pytest.approx(1.0)
        assert stat['m2'] == pytest.approx(0.0)

        total_count += stat['count']

    assert total_count == img.size - 999


def test_build_data_block_label_canonicalize_base_layer():
    '''
    Given: Label inputs without reclassification mappings.
    When: Building a DataBlock.
    Then: Compile the single base label channel in the stack.
    '''
    raster_data = _make_raster_data(
        label_array=BASE_LABEL_ARRAY,
        label_specs=BASE_LABELSPECS,
    )
    block = builder.build_data_block('test_block', raster_data)

    assert len(block.data.label) == 1
    assert block.manifest['label_num_cls'] == {'base': 4}
    assert block.manifest['label_ignore_cls'] == {'base': [4]}
    assert block.manifest['label_cls_names'] == {
        'base': ['WAT', 'FOR', 'WET', 'UCL']
    }


def test_build_data_block_label_canonicalize_zero_based_index():
    '''
    Given: 0-based label array (0, 1, 2, 3) and index_base=0 specs.
    When: Building a DataBlock.
    Then: Shift base layer to canonical 1-based indexing (1, 2, 3, 4)
        and map names correctly.
    '''
    zero_based_array = numpy.repeat([0, 1, 2, 3], 16384).reshape((1, 256, 256))
    zero_based_specs: dict[str, geo_core.CategoricalSpec] = {
        'base': {
            'num_cls': 4,
            'ignore_cls': (3,),
            'index_base': 0,
            'class_name': {'0': 'WAT', '1': 'FOR', '2': 'WET', '3': 'UCL'},
        }
    }
    raster_data = _make_raster_data(
        label_array=zero_based_array,
        label_specs=zero_based_specs,
    )
    block = builder.build_data_block('test_block', raster_data)

    stack_base = block.data.label[0]
    assert set(numpy.unique(stack_base)) == {1, 2, 3, 255}
    assert block.manifest['label_num_cls'] == {'base': 4}
    assert block.manifest['label_ignore_cls'] == {'base': [3]}
    assert block.manifest['label_cls_names'] == {
        'base': ['WAT', 'FOR', 'WET', 'UCL']
    }
    assert block.manifest['label_count']['base'] == [16384, 16384, 16384, 0]


def test_build_data_block_label_stats_valid_ratio():
    '''
    Given: A label array with ignore categories.
    When: Building a DataBlock.
    Then: Calculate the valid pixel ratio correctly.
    '''
    raster_data = _make_raster_data(
        label_array=BASE_LABEL_ARRAY,
        label_specs=BASE_LABELSPECS,
    )
    block = builder.build_data_block('test_block', raster_data)

    valid = float(numpy.mean(BASE_LABEL_ARRAY != 4))
    assert block.manifest['valid_ratios'].get('base') == pytest.approx(valid)


def test_build_data_block_label_stats_class_count_entropy():
    '''
    Given: A label array.
    When: Building a DataBlock.
    Then: Calculate absolute class counts and Shannon entropy.
    '''
    raster_data = _make_raster_data(
        label_array=BASE_LABEL_ARRAY,
        label_specs=BASE_LABELSPECS,
    )
    block = builder.build_data_block('test_block', raster_data)

    ent = pytest.approx(- numpy.log2(1 / 3))
    assert block.manifest['label_count'] == {'base': [16384, 16384, 16384, 0]}
    assert block.manifest['label_entropy'] == {'base': ent}


def test_inputs_post_init_label_specs_missing():
    '''
    Given: A label array but no label specs.
    When: Building a DataBlock with labels but no specs.
    Then: Raise an AssertionError.
    '''
    raster_data = _make_raster_data(
        label_array=BASE_LABEL_ARRAY,
        label_specs=None,
    )
    with pytest.raises(AssertionError):
        builder.build_data_block('test_block', raster_data)


def test_inputs_property_pad_dem_raise_when_not_provided():
    '''
    Given: A RasterReadOutput instance with no padded DEM array.
    When: Accessing the pad_dem property.
    Then: Raise a ValueError.
    '''
    raster_data = _make_raster_data(image_padded_dem=None)
    with pytest.raises(ValueError, match='Cannot access padded DEM'):
        _ = raster_data.pad_dem


# ----- `validate_feature_specs` tests
def test_config_post_init_invalid_spectral_indices():
    '''
    Given: An invalid index name.
    When: Validating feature specs on context.
    Then: Raise a ValueError.
    '''
    ctx = _make_context(band_map={'red': 0})
    with pytest.raises(ValueError, match='Invalid spectral indices'):
        ctx.validate_feature_specs(add_spectral=['foo'])


def test_config_post_init_missing_red_for_any_spectral():
    '''
    Given: A context requesting ndvi but lacking red band mappings.
    When: Validating feature specs.
    Then: Raise a ValueError.
    '''
    ctx = _make_context(band_map={'foo': 0})
    with pytest.raises(ValueError, match='red band missing'):
        ctx.validate_feature_specs(add_spectral=['ndvi'])


@pytest.mark.parametrize(
    'indice, required',
    [('ndvi', 'NIR'), ('ndmi', 'SWIR1'), ('nbr', 'SWIR2')]
)
def test_config_post_init_missing_required_bands(indice, required):
    '''
    Given: A context requesting a spectral index without required bands.
    When: Validating feature specs.
    Then: Raise a ValueError.
    '''
    ctx = _make_context(band_map={'red': 0})
    with pytest.raises(ValueError, match=f'{required} band missing'):
        ctx.validate_feature_specs(add_spectral=[indice])


def test_validate_block_feature_specs_valid():
    '''
    Given: Valid band names, spectral indices, and topo features.
    When: Validating block feature specs.
    Then: Validation passes without raising any exceptions.
    '''
    bands = ['red', 'green', 'blue', 'nir', 'swir1', 'swir2', 'dem']
    ctx = _make_context(
        band_map={b: i for i, b in enumerate(bands)},
        label_fpath=pathlib.Path('lbl.tif'),
        label_specs=BASE_LABELSPECS,
    )
    ctx.validate_feature_specs(
        add_spectral=['ndvi', 'ndmi', 'nbr'],
        add_topo=['slope', 'aspect', 'tpi'],
    )


def test_validate_block_feature_specs_missing_label_specs():
    '''
    Given: label_fpath provided but no label_specs.
    When: Validating block feature specs.
    Then: Raise a ValueError indicating label specs are missing.
    '''
    ctx = _make_context(
        band_map={'red': 0},
        label_fpath=pathlib.Path('lbl.tif'),
        label_specs=None,
    )
    with pytest.raises(ValueError, match='"label_specs" not provided'):
        ctx.validate_feature_specs()


def test_validate_block_feature_specs_invalid_features():
    '''
    Given: Unknown spectral or topographic features.
    When: Validating block feature specs.
    Then: Raise a ValueError.
    '''
    ctx = _make_context(band_map={'red': 0})
    with pytest.raises(ValueError, match='Invalid spectral indices'):
        ctx.validate_feature_specs(add_spectral=['unknown'])
    with pytest.raises(ValueError, match='Invalid topo features'):
        ctx.validate_feature_specs(add_topo=['unknown'])


# ----- `_Calc` tests
def test_calc_slope_and_aspect():
    '''
    Given: Padded 3x3 DEM window.
    When: Computing slope and aspect.
    Then: Output matches trigonometric gradients.
    '''
    # plane sloping east: z = x (dz/dx = 1, dz/dy = 0)
    flat = numpy.array([
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0]
    ])
    slope, cos_a, sin_a = builder._Calc.slope_n_aspect(flat, -999.9)

    # slope gradient magnitude is 1.0
    assert slope == pytest.approx(1.0, rel=1e-3)
    assert cos_a == pytest.approx(-1.0, abs=1e-3)
    assert sin_a == pytest.approx(0.0, abs=1e-3)

    # nodata handling: invalid center
    flat[1, 1] = numpy.nan
    s, c, sn = builder._Calc.slope_n_aspect(flat, -999.9)
    assert s == -999.9
    assert c == -999.9
    assert sn == -999.9


def test_calc_tpi():
    '''
    Given: Padded (2k-1)x(2k-1) DEM window.
    When: Computing Topographic Position Index (TPI).
    Then: Center pixel minus mean of neighbors.
    '''
    # 3x3 window: center is 10, mean of all 9 is (8*20 + 10)/9 = 170/9 = 18.888
    pit = numpy.full((3, 3), 20.0)
    pit[1, 1] = 10.0
    assert builder._Calc.tpi(pit, -999.9) == pytest.approx(-10.0)

    flat = numpy.full((3, 3), 42.0)
    assert builder._Calc.tpi(flat, -999.9) == pytest.approx(0.0)

    # invalid center cell returns nodata
    pit[1, 1] = numpy.nan
    assert builder._Calc.tpi(pit, -999.9) == -999.9


# ----- private helpers
def _make_raster_data(**overrides):
    base = io.RasterReadOutput(
        image_array=numpy.ones((7, 256, 256), dtype=numpy.float32),
        image_nodata=numpy.nan,
        image_band_map={
            'red': 0,
            'green': 1,
            'blue': 2,
            'nir': 3,
            'swir1': 4,
            'swir2': 5,
            'dem': 6,
        },
        image_padded_dem=None,
        label_array=None,
        label_nodata=0,
        label_specs=None,
    )
    return dataclasses.replace(base, **overrides)


def _make_context(
    band_map: dict[str, int] | None = None,
    label_specs: dict[str, geo_core.CategoricalSpec] | None = None,
    label_fpath: pathlib.Path | None = None,
) -> io.RasterReadContext:
    return io.RasterReadContext(
        image_fpath=pathlib.Path('img.tif'),
        label_fpath=label_fpath,
        image_band_map=band_map or {'red': 0},
        label_specs=label_specs,
    )


def _is_invalid(arr, nodata):
    if isinstance(nodata, float) and numpy.isnan(nodata):
        return numpy.isnan(arr)
    return arr == nodata
