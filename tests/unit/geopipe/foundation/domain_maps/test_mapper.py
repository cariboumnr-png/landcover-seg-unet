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

# pylint: disable=protected-access

'''Unit tests for domain mapper (mapper.py).'''

# third-party imports
import numpy
import pytest
import rasterio
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.domain_maps.mapper as domain_mapper


# ----- `map_domain_to_grid` tests
def test_map_domain_to_grid_success(dummy_geotiff_factory):
    '''
    Given: A dummy categorical GeoTIFF with labels [10, 11]
        and index_base=10.
    When: `map_domain_to_grid` is executed.
    Then: Remap labels to zero-based range [0, 1] and return
        tile dictionary package.
    '''
    raster_path = str(dummy_geotiff_factory('categorical.tif', 16, 16, 1, dtype=numpy.int16))

    # override raster content to contain explicit integer categories 10 and 11
    # write to band 1
    with rasterio.open(raster_path, 'r+') as dst:
        nodata_val = int(dst.nodata) if dst.nodata is not None else -1
        arr = dst.read(1)
        arr[...] = 10
        arr[0:8, 0:8] = 11
        arr[8:16, 8:16] = nodata_val
        dst.write(arr, 1)
        file_crs = dst.crs.to_string()

    spec = geo_core.GridSpec(
        crs=file_crs,
        origin=(0.5, 0.5), # top-left origin matching dummy raster
        pixel_size=(1.0, 1.0),
        tile_size=(8, 8),
        tile_overlap=(0, 0),
        grid_shape=(2, 2)
    )
    grid = geo_core.GridLayout(mode='tiles', spec=spec)

    result = domain_mapper.map_domain_to_grid(grid, raster_path, index_base=10)

    # checks
    assert len(result) == 5 # 4 grid tiles + (-999, -999) max_idx marker
    # tile (0, 0) is all 11 -> remapped to index 1
    assert numpy.all(result[(0, 0)] == 1)
    # tile (0, 8) is all 10 -> remapped to index 0
    assert numpy.all(result[(0, 8)] == 0)
    # tile (8, 8) is all nodata -> remapped to -1
    assert numpy.all(result[(8, 8)] == -1)

    # verify the max index marker entry
    assert numpy.all(result[(-999, -999)] == 1) # max category index is 1


def test_map_domain_to_grid_wrong_base(dummy_geotiff_factory):
    '''
    Given: A dummy categorical GeoTIFF with label values [10, 11].
    When: Running `map_domain_to_grid` with expected `index_base=11`.
    Then: Raise a ValueError indicating base mismatch.
    '''
    raster_path = str(dummy_geotiff_factory('categorical.tif', 16, 16, 1, dtype=numpy.int16))
    with rasterio.open(raster_path, 'r+') as dst:
        arr = dst.read(1)
        arr[...] = 10
        dst.write(arr, 1)
        file_crs = dst.crs.to_string()

    spec = geo_core.GridSpec(
        crs=file_crs,
        origin=(0.5, 0.5),
        pixel_size=(1.0, 1.0),
        tile_size=(8, 8),
        tile_overlap=(0, 0),
        grid_shape=(2, 2)
    )
    grid = geo_core.GridLayout(mode='tiles', spec=spec)

    with pytest.raises(ValueError, match='Min value 10 != base 11'):
        domain_mapper.map_domain_to_grid(grid, raster_path, index_base=11)


def test_map_domain_to_grid_empty(dummy_geotiff_factory):
    '''
    Given: A dummy categorical GeoTIFF with all values equal
        to nodata.
    When: Running `map_domain_to_grid`.
    Then: Raise a ValueError due to lack of valid categories.
    '''
    raster_path = str(dummy_geotiff_factory('categorical.tif', 16, 16, 1, dtype=numpy.int16))
    with rasterio.open(raster_path, 'r+') as dst:
        nodata_val = int(dst.nodata) if dst.nodata is not None else -1
        arr = dst.read(1)
        arr[...] = nodata_val
        dst.write(arr, 1)
        file_crs = dst.crs.to_string()

    spec = geo_core.GridSpec(
        crs=file_crs,
        origin=(0.5, 0.5),
        pixel_size=(1.0, 1.0),
        tile_size=(8, 8),
        tile_overlap=(0, 0),
        grid_shape=(2, 2)
    )
    grid = geo_core.GridLayout(mode='tiles', spec=spec)

    with pytest.raises(ValueError, match='No valid data values found'):
        domain_mapper.map_domain_to_grid(grid, raster_path, index_base=10)
