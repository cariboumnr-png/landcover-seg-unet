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

'''Unit tests for spatial grid warping and resampling algorithms.'''

# third-party imports
import numpy
import rasterio
import rasterio.crs
import rasterio.enums
import rasterio.transform
# local imports
import landseg.geopipe.rasters.warp as warp


# ----- `resolve_resampling_algorithm` tests
def test_resolve_resampling_algorithm():
    '''
    Given: Categorical flags and algorithm names.
    When: Resolving algorithm enum through `resolve_resampling_algorithm`.
    Then: Maps categorical to nearest and resolves named algorithms.
    '''
    # categorical always forces nearest
    res = warp.resolve_resampling_algorithm('bilinear', is_categorical=True)
    assert res == rasterio.enums.Resampling.nearest

    res = warp.resolve_resampling_algorithm('nearest', is_categorical=False)
    assert res == rasterio.enums.Resampling.nearest

    res = warp.resolve_resampling_algorithm('cubic', is_categorical=False)
    assert res == rasterio.enums.Resampling.cubic

    res = warp.resolve_resampling_algorithm('lanczos', is_categorical=False)
    assert res == rasterio.enums.Resampling.lanczos

    # invalid algorithm name defaults to bilinear
    res = warp.resolve_resampling_algorithm('invalid', is_categorical=False)
    assert res == rasterio.enums.Resampling.bilinear


# ----- `write_warped_vrt` tests
def test_write_warped_vrt_multiband(dummy_geotiff_factory, tmp_path):
    '''
    Given: A 10-band Sentinel-2 composite in UTM Zone 17N (EPSG:32617).
    When: Warping to EPSG:3161 at 20m resolution using `write_warped_vrt`.
    Then: Output VRT has 10 bands in EPSG:3161 with exact dimensions.
    '''
    s2_path = dummy_geotiff_factory(
        filename='sentinel2_10band_utm17.tif',
        width=50,
        height=50,
        bands=10,
        crs='EPSG:32617',
        dtype=numpy.uint16,
    )
    dst_crs = rasterio.crs.CRS.from_string('EPSG:3161')
    dst_transform = rasterio.transform.from_origin(
        500000.0, 6000000.0, 20.0, 20.0
    )
    out_path = tmp_path / 'warped_s2.vrt'

    with rasterio.open(s2_path) as src:
        result_path = warp.write_warped_vrt(
            src=src,
            output_path=out_path,
            crs=dst_crs,
            transform=dst_transform,
            width=50,
            height=50,
            resampling=rasterio.enums.Resampling.bilinear,
        )

    assert result_path == out_path
    assert out_path.exists()

    with rasterio.open(out_path) as dst:
        assert dst.crs.to_string() == 'EPSG:3161'
        assert dst.count == 10
        assert dst.width == 50
        assert dst.height == 50


def test_write_warped_vrt_categorical_preserves_integers(
    dummy_geotiff_factory,
    tmp_path,
):
    '''
    Given: A 5-class integer landcover label mask in EPSG:32617.
    When: Warping with nearest-neighbor categorical resampling.
    Then: Output VRT contains strictly integer class IDs.
    '''
    def _label_gen(shape, _):
        return numpy.random.randint(1, 6, shape).astype(numpy.uint8)

    label_path = dummy_geotiff_factory(
        filename='labels_5class_utm17.tif',
        width=50,
        height=50,
        bands=1,
        crs='EPSG:32617',
        dtype=numpy.uint8,
        data_gen_func=_label_gen,
    )
    dst_crs = rasterio.crs.CRS.from_string('EPSG:3161')
    dst_transform = rasterio.transform.from_origin(
        500000.0, 6000000.0, 20.0, 20.0
    )
    out_path = tmp_path / 'warped_labels.vrt'

    with rasterio.open(label_path) as src:
        warp.write_warped_vrt(
            src=src,
            output_path=out_path,
            crs=dst_crs,
            transform=dst_transform,
            width=50,
            height=50,
            resampling=rasterio.enums.Resampling.nearest,
        )

    with rasterio.open(out_path) as dst:
        assert dst.crs.to_string() == 'EPSG:3161'
        assert dst.count == 1
        assert dst.width == 50
        assert dst.height == 50
        data = dst.read(1)
        unique_vals = numpy.unique(data)
        assert all(
            val in range(1, 6) or val in (0, 255) for val in unique_vals
        )
    assert data.dtype == numpy.uint8


def test_write_warped_vrt_nodata_and_dtype(dummy_geotiff_factory, tmp_path):
    '''
    Given: A GeoTIFF source raster.
    When: Specifying explicit nodata and dtype parameters to write_warped_vrt.
    Then: Output VRT reflects the custom nodata value and data type.
    '''
    src_path = dummy_geotiff_factory(
        filename='nodata_src.tif',
        width=10,
        height=10,
        bands=1,
        dtype=numpy.int16,
    )
    dst_crs = rasterio.crs.CRS.from_string('EPSG:3161')
    dst_transform = rasterio.transform.from_origin(
        500000.0, 6000000.0, 20.0, 20.0
    )
    out_path = tmp_path / 'warped_nodata.vrt'

    with rasterio.open(src_path) as src:
        warp.write_warped_vrt(
            src=src,
            output_path=out_path,
            crs=dst_crs,
            transform=dst_transform,
            width=10,
            height=10,
            resampling=rasterio.enums.Resampling.nearest,
            nodata=-9999.0,
            dtype='float32',
        )

    with rasterio.open(out_path) as dst:
        assert dst.nodata == -9999.0
        assert dst.dtypes[0] == 'float32'


def test_write_warped_vrt_from_path(dummy_geotiff_factory, tmp_path):
    '''
    Given: A path to a source GeoTIFF.
    When: Calling `write_warped_vrt` with a Path instead of DatasetReader.
    Then: Automatically open, warp, and produce valid warped VRT.
    '''
    src_path = dummy_geotiff_factory(
        filename='path_src.tif',
        width=10,
        height=10,
        bands=1,
    )
    dst_crs = rasterio.crs.CRS.from_string('EPSG:3161')
    dst_transform = rasterio.transform.from_origin(
        500000.0, 6000000.0, 20.0, 20.0
    )
    out_path = tmp_path / 'warped_from_path.vrt'

    res = warp.write_warped_vrt(
        src=src_path,
        output_path=out_path,
        crs=dst_crs,
        transform=dst_transform,
        width=10,
        height=10,
        resampling=rasterio.enums.Resampling.nearest,
    )
    assert res == out_path
    assert out_path.exists()


# ----- `is_same_crs` tests
def test_is_same_crs():
    '''
    Given: Identical, differing, or None CRS inputs.
    When: Checking equivalence through `is_same_crs`.
    Then: Return boolean reflecting CRS equivalence.
    '''
    assert warp.is_same_crs('EPSG:3161', 'EPSG:3161')
    assert not warp.is_same_crs('EPSG:3161', 'EPSG:4326')
    assert warp.is_same_crs(None, None)
    assert not warp.is_same_crs('EPSG:3161', None)
    assert not warp.is_same_crs(None, 'EPSG:3161')
    assert warp.is_same_crs('custom_crs', 'custom_crs')
