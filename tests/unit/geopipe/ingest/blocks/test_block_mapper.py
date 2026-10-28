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

'''Unit tests for raster-to-grid mapper and alignment validation.'''

# standard imports
import pathlib
# third-party imports
import pytest
# local imports
import landseg.geopipe.grid.handlers as grid_handlers
import landseg.geopipe.ingest.blocks.mapper as mapper
import landseg.geopipe.ingest.blocks.mapper.mapper as raw_mapper
import landseg.geopipe.rasters as rasters


# ----- `validate_raster_alignment` tests
def test_validate_raster_alignment_matching_rasters(dummy_geotiff_factory):
    '''
    Given: An image and label raster sharing CRS and resolution.
    When: `validate_raster_alignment` is called against the world grid.
    Then: Return pixel resolution tuple and intersected bounding box.
    '''
    ref_fp = dummy_geotiff_factory(
        filename='ref.tif', width=32, height=32, crs='EPSG:32617'
    )
    img_fp = dummy_geotiff_factory(
        filename='img.tif', width=32, height=32, crs='EPSG:32617'
    )
    lbl_fp = dummy_geotiff_factory(
        filename='lbl.tif', width=32, height=32, crs='EPSG:32617'
    )
    grid = grid_handlers.validate_grid_inputs(
        tile_size=(16, 16),
        tile_stride=(8, 8),
        mode='ref',
        ref_fpath=pathlib.Path(ref_fp),
    )

    with rasters.read_rasters(
        pathlib.Path(img_fp), pathlib.Path(lbl_fp)
    ) as (img, lbl):
        pixel_size, bbox = mapper.validate_raster_alignment(grid, img, lbl)

    assert len(pixel_size) == 2
    assert bbox.left < bbox.right
    assert bbox.bottom < bbox.top


def test_validate_raster_alignment_crs_mismatch(dummy_geotiff_factory):
    '''
    Given: An image raster with CRS different from the world grid.
    When: `validate_raster_alignment` is called.
    Then: Raise a ValueError indicating CRS mismatch.
    '''
    ref_fp = dummy_geotiff_factory(
        filename='ref.tif', width=32, height=32, crs='EPSG:32617'
    )
    img_fp = dummy_geotiff_factory(
        filename='img_mismatch.tif', width=32, height=32, crs='EPSG:32618'
    )
    grid = grid_handlers.validate_grid_inputs(
        tile_size=(16, 16),
        tile_stride=(8, 8),
        mode='ref',
        ref_fpath=pathlib.Path(ref_fp),
    )

    with rasters.read_rasters(pathlib.Path(img_fp), None) as (img, _):
        with pytest.raises(ValueError, match='Data CRS .* != world CRS'):
            mapper.validate_raster_alignment(grid, img)


def test_validate_raster_alignment_missing_image(dummy_geotiff_factory):
    '''
    Given: A missing image dataset reader.
    When: `validate_raster_alignment` is called with img=None.
    Then: Raise a ValueError.
    '''
    ref_fp = dummy_geotiff_factory(
        filename='ref.tif', width=32, height=32, crs='EPSG:32617'
    )
    grid = grid_handlers.validate_grid_inputs(
        tile_size=(16, 16),
        tile_stride=(8, 8),
        mode='ref',
        ref_fpath=pathlib.Path(ref_fp),
    )

    with pytest.raises(ValueError, match='A valid image raster is required'):
        mapper.validate_raster_alignment(grid, None)


# ----- `map_rasters` tests
def test_map_rasters_success(dummy_geotiff_factory):
    '''
    Given: Valid image raster aligned with the target world grid.
    When: `map_rasters` is executed.
    Then: Return MappedRasterWindows containing windows for tiles.
    '''
    ref_fp = dummy_geotiff_factory(
        filename='ref.tif', width=32, height=32, crs='EPSG:32617'
    )
    img_fp = dummy_geotiff_factory(
        filename='img.tif', width=32, height=32, crs='EPSG:32617'
    )
    grid = grid_handlers.validate_grid_inputs(
        tile_size=(16, 16),
        tile_stride=(8, 8),
        mode='ref',
        ref_fpath=pathlib.Path(ref_fp),
    )

    windows = raw_mapper.map_rasters(grid, pathlib.Path(img_fp), None)

    assert windows.grid_id == grid.gid
    assert len(windows.image) > 0
    assert len(windows.label) == 0
