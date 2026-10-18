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

'''Unit tests for spatial geometry definition and raster warping.'''

# third-party imports
import numpy
import rasterio
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.harmonize.rasters.spatial as spatial


# ----- `warp_to_grid` tests
def test_warp_sentinel2_to_grid(dummy_geotiff_factory, tmp_path):
    '''
    Given: A 10-band Sentinel-2 composite in UTM Zone 17N (EPSG:32617).
    When: Warping to a target GridLayout in EPSG:3161 at 20m resolution.
    Then: Output VRT has 10 bands in EPSG:3161 with exact grid dimensions.
    '''
    s2_path = dummy_geotiff_factory(
        filename='sentinel2_10band_utm17.tif',
        width=50,
        height=50,
        bands=10,
        crs='EPSG:32617',
        dtype=numpy.uint16
    )
    spec = geo_core.GridSpec(
        origin=(500000.0, 601000.0),
        pixel_size=(20.0, 20.0),
        tile_size=(25, 25),
        tile_stride=(0, 0),
        grid_extent=(1000.0, 1000.0),
        crs='EPSG:3161'
    )
    layout = geo_core.GridLayout(spec)

    out_path = tmp_path / 'warped_s2.vrt'
    spatial.warp_to_grid(
        input_path=s2_path,
        output_path=out_path,
        world_grid=layout,
        is_categorical=False
    )

    with rasterio.open(out_path) as dst:
        assert dst.crs.to_string() == 'EPSG:3161'
        assert dst.count == 10
        assert dst.width == 50
        assert dst.height == 50


def test_warp_categorical_labels_preserves_integer_values(
    dummy_geotiff_factory,
    tmp_path
):
    '''
    Given: A 5-class integer landcover label mask in EPSG:32617.
    When: Warping using nearest-neighbor categorical resampling.
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
        data_gen_func=_label_gen
    )
    spec = geo_core.GridSpec(
        origin=(500000.0, 601000.0),
        pixel_size=(20.0, 20.0),
        tile_size=(25, 25),
        tile_stride=(0, 0),
        grid_extent=(1000.0, 1000.0),
        crs='EPSG:3161'
    )
    layout = geo_core.GridLayout(spec)

    out_path = tmp_path / 'warped_labels.vrt'
    spatial.warp_to_grid(
        input_path=label_path,
        output_path=out_path,
        world_grid=layout,
        is_categorical=True
    )

    with rasterio.open(out_path) as dst:
        assert dst.crs.to_string() == 'EPSG:3161'
        assert dst.count == 1
        assert dst.width == 50
        assert dst.height == 50
        data = dst.read(1)
        unique_vals = numpy.unique(data)
        assert all(val in range(1, 6) or val in (0, 255) for val in unique_vals)
    assert data.dtype == numpy.uint8
