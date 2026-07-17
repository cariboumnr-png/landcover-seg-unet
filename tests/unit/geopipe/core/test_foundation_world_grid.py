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

'''Unit tests for world-grid tiling utility (foundation_world_grid.py).'''

# third-party imports
import pytest
import rasterio.windows
# local imports
import landseg.geopipe.core.foundation_world_grid as world_grid


# ----- `GridSpec` tests
def test_gridspec_validation():
    '''
    Given: An invalid overlap specification.
    When: `GridSpec` is initialized.
    Then: Raise a ValueError in post-init.
    '''
    with pytest.raises(ValueError, match='Overlap must be smaller'):
        world_grid.GridSpec(
            crs='EPSG:32617',
            origin=(500000.0, 5000000.0),
            pixel_size=(10.0, 10.0),
            tile_size=(256, 256),
            tile_overlap=(300, 128)
        )


# ----- `GridLayout` tests
def test_gridlayout_bbox_generation():
    '''
    Given: A GridSpec with spatial extent.
    When: `GridLayout` is constructed in `'bbox'` mode.
    Then: Correct row/col tiles are generated within extent boundary
        constraints.
    '''
    spec = world_grid.GridSpec(
        crs='EPSG:32617',
        origin=(0.0, 1000.0),
        pixel_size=(10.0, 10.0),
        tile_size=(256, 256),
        tile_overlap=(128, 128),
        grid_extent=(5120.0, 5120.0) # 512x512 pixels
    )

    layout = world_grid.GridLayout(mode='bbox', spec=spec)

    # 512x512 grid, tile size 256, step size = 256-128 = 128.
    # coordinates row/col step ranges:
    # row in range(0, 512, 128) -> [0, 128, 256, 384]
    # col in range(0, 512, 128) -> [0, 128, 256, 384]
    # total tiles = 4 * 4 = 16
    assert len(layout) == 16
    assert (0, 0) in layout
    assert (384, 384) in layout

    # check boundaries on the edge tile
    edge_tile = layout[(384, 384)]
    assert edge_tile.col_off == 384
    assert edge_tile.row_off == 384
    assert edge_tile.width == 128 # min(256, 512-384)
    assert edge_tile.height == 128 # min(256, 512-384)


def test_gridlayout_tiles_generation():
    '''
    Given: A GridSpec with a target grid shape.
    When: `GridLayout` is constructed in `'tiles'` mode.
    Then: Generate fixed tile count exactly matching requested shape.
    '''
    spec = world_grid.GridSpec(
        crs='EPSG:32617',
        origin=(0.0, 1000.0),
        pixel_size=(10.0, 10.0),
        tile_size=(256, 256),
        tile_overlap=(128, 128),
        grid_shape=(3, 3)
    )

    layout = world_grid.GridLayout(mode='tiles', spec=spec)

    assert len(layout) == 9
    assert layout.extent == (512, 512) # (3-1)*128 + 256 = 512


def test_gridlayout_container_protocol():
    '''
    Given: A constructed `GridLayout`.
    When: Running dictionary protocol checks.
    Then: Iteration, length, keys, index validation, and value type
        checks succeed.
    '''
    spec = world_grid.GridSpec(
        crs='EPSG:32617',
        origin=(0.0, 1000.0),
        pixel_size=(10.0, 10.0),
        tile_size=(256, 256),
        tile_overlap=(128, 128),
        grid_shape=(1, 1)
    )
    layout = world_grid.GridLayout(mode='tiles', spec=spec)

    assert len(layout) == 1
    assert list(layout.keys()) == [(0, 0)]
    assert isinstance(layout[(0, 0)], rasterio.windows.Window)

    with pytest.raises(TypeError, match='Index must be \\(x, y\\) in pixels'):
        _ = layout['invalid'] # type: ignore

    with pytest.raises(TypeError, match='Index must be \\(x, y\\) in pixels'):
        _ = layout[(0.5, 0)] # type: ignore


def test_gridlayout_offset_alignment():
    '''
    Given: An Affine transform describing an offset raster.
    When: Computing offsets via `offset_from`.
    Then: Shift the output windows by the computed pixel offset amount.
    '''
    spec = world_grid.GridSpec(
        crs='EPSG:32617',
        origin=(500000.0, 5000000.0),
        pixel_size=(10.0, 10.0),
        tile_size=(256, 256),
        tile_overlap=(128, 128),
        grid_shape=(1, 1)
    )
    layout = world_grid.GridLayout(mode='tiles', spec=spec)

    # raster is shifted right by 10 pixels and down by 20 pixels
    # affine transformation: a = 10.0, b = 0.0, c = rx = 500100.0
    # d = 0.0, e = -10.0, f = ry = 4999800.0
    transform = rasterio.Affine(10.0, 0.0, 500100.0, 0.0, -10.0, 4999800.0)

    layout.offset_from(transform)

    # original window was at (0, 0, 256, 256)
    # col offset shift: dc = (500100.0 - 500000.0)/10 = 10
    # row offset shift: dr = (5000000.0 - 4999800.0)/10 = 20
    # shifted window: xoff = 0 - 10 = -10, yoff = 0 - 20 = -20
    shifted_window = layout[(0, 0)]
    assert shifted_window.col_off == -10
    assert shifted_window.row_off == -20


def test_gridlayout_serialization_roundtrip():
    '''
    Given: A generated `GridLayout`.
    When: Reconstructed using `to_payload` and `from_payload`.
    Then: All state attributes are perfectly restored.
    '''
    spec = world_grid.GridSpec(
        crs='EPSG:32617',
        origin=(0.0, 1000.0),
        pixel_size=(10.0, 10.0),
        tile_size=(256, 256),
        tile_overlap=(128, 128),
        grid_shape=(2, 2)
    )
    layout = world_grid.GridLayout(mode='tiles', spec=spec)

    payload = layout.to_payload()
    restored = world_grid.GridLayout.from_payload(payload)

    assert restored._mode == layout._mode
    assert restored._extent == layout._extent
    assert len(restored) == len(layout)
    assert restored[(0, 0)] == layout[(0, 0)]
