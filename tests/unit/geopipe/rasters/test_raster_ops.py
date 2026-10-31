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

'''Unit tests for raster composition and mask unification.'''

# third-party imports
import numpy
import pytest
import rasterio
# local imports
import landseg.geopipe.rasters.mask as mask
import landseg.geopipe.rasters.stack as stack


# ----- `stack_rasters` tests
def test_stack_rasters(dummy_geotiff_factory, tmp_path):
    '''
    Given: A 10-band optical TIFF and a 1-band DEM TIFF aligned in EPSG:3161.
    When: Invoking `stack_rasters`.
    Then: Produces an 11-band composite Virtual Raster (.vrt).
    '''
    opt_path = dummy_geotiff_factory(
        filename='aligned_opt.tif',
        width=20,
        height=20,
        bands=10,
        dtype=numpy.uint16,
    )
    dem_path = dummy_geotiff_factory(
        filename='aligned_dem.tif',
        width=20,
        height=20,
        bands=1,
        dtype=numpy.float32,
    )

    out_vrt = tmp_path / 'composite.vrt'
    out_composite = stack.stack_rasters([opt_path, dem_path], out_vrt)

    assert out_composite is not None

    with rasterio.open(out_composite) as dst:
        assert dst.count == 11
        assert dst.crs.to_string() == 'EPSG:3161'
        assert dst.width == 20
        assert dst.height == 20


def test_stack_rasters_empty_raises(tmp_path):
    '''
    Given: An empty list of source paths.
    When: Invoking `stack_rasters`.
    Then: Raise ValueError.
    '''
    with pytest.raises(ValueError, match='source_paths list cannot be empty'):
        stack.stack_rasters([], tmp_path / 'empty.vrt')


def test_stack_rasters_mismatched_dimensions_raises(
    dummy_geotiff_factory, tmp_path
):
    '''
    Given: Rasters with differing dimensions.
    When: Invoking `stack_rasters`.
    Then: Raise ValueError.
    '''
    p1 = dummy_geotiff_factory(filename='s1.tif', width=10, height=10)
    p2 = dummy_geotiff_factory(filename='s2.tif', width=20, height=20)

    with pytest.raises(ValueError, match='different dimensions'):
        stack.stack_rasters([p1, p2], tmp_path / 'dim.vrt')


def test_stack_rasters_mismatched_crs_raises(
    dummy_geotiff_factory, tmp_path
):
    '''
    Given: Rasters with differing CRS definitions.
    When: Invoking `stack_rasters`.
    Then: Raise ValueError.
    '''
    p1 = dummy_geotiff_factory(
        filename='c1.tif', width=10, height=10, crs='EPSG:3161'
    )
    p2 = dummy_geotiff_factory(
        filename='c2.tif', width=10, height=10, crs='EPSG:4326'
    )

    with pytest.raises(ValueError, match='different CRS'):
        stack.stack_rasters([p1, p2], tmp_path / 'crs.vrt')


def test_stack_rasters_band_tags_mismatch_raises(
    dummy_geotiff_factory, tmp_path
):
    '''
    Given: Band tags count that does not match total source bands.
    When: Invoking `stack_rasters`.
    Then: Raise ValueError.
    '''
    p1 = dummy_geotiff_factory(filename='bt1.tif', width=10, height=10, bands=2)

    with pytest.raises(ValueError, match='band_tags length must equal'):
        stack.stack_rasters(
            [p1],
            tmp_path / 'tags_err.vrt',
            band_tags=[{'name': 'only_one'}],
        )


def test_stack_rasters_with_metadata(dummy_geotiff_factory, tmp_path):
    '''
    Given: Source raster with custom band and dataset metadata tags.
    When: Invoking `stack_rasters`.
    Then: Produces composite with attached XML metadata tags.
    '''
    p1 = dummy_geotiff_factory(
        filename='meta_src.tif', width=10, height=10, bands=2
    )
    out_vrt = tmp_path / 'meta.vrt'
    stack.stack_rasters(
        [p1],
        out_vrt,
        band_tags=[{'channel': 'blue'}, {'channel': 'green'}],
        dataset_tags={'sensor': 'custom_sensor'},
    )

    with rasterio.open(out_vrt) as src:
        assert src.tags().get('sensor') == 'custom_sensor'
        assert src.tags(1).get('channel') == 'blue'
        assert src.tags(2).get('channel') == 'green'


# ----- `unify_nodata_mask` tests
def test_unify_nodata_mask(dummy_geotiff_factory, tmp_path):
    '''
    Given: A multi-band composite GeoTIFF with nodata flags.
    When: Invoking `unify_nodata_mask`.
    Then: Creates a 1-band boolean valid pixel mask Virtual Raster (.vrt).
    '''
    comp_path = dummy_geotiff_factory(
        filename='comp_with_nodata.tif',
        width=20,
        height=20,
        bands=3,
        dtype=numpy.uint8,
    )

    out_mask = tmp_path / 'valid_mask.vrt'
    mask.unify_nodata_mask(comp_path, out_mask)

    with rasterio.open(out_mask) as dst:
        assert dst.count == 1
        assert dst.dtypes[0] == 'uint8'
        assert dst.width == 20
        assert dst.height == 20


def test_unify_nodata_mask_gtiff_output(dummy_geotiff_factory, tmp_path):
    '''
    Given: A multi-band composite GeoTIFF.
    When: Invoking `unify_nodata_mask` with a .tif destination path.
    Then: Creates a 1-band GeoTIFF mask file directly.
    '''
    comp_path = dummy_geotiff_factory(
        filename='comp_gtiff.tif',
        width=20,
        height=20,
        bands=2,
        dtype=numpy.uint8,
    )

    out_mask = tmp_path / 'valid_mask.tif'
    mask.unify_nodata_mask(comp_path, out_mask)

    with rasterio.open(out_mask) as dst:
        assert dst.count == 1
        assert dst.driver == 'GTiff'
        assert dst.width == 20
        assert dst.height == 20
