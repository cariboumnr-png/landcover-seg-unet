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

# pylint: disable=protected-access

'''Unit tests for world-grid tiling utility (grid_layout.py).'''

# standard imports
import json
import os
# third-party imports
import pytest
import rasterio.windows
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.contracts as contracts
import landseg.geopipe.core.grid_layout as grid_layout


# ----- `GridSpec` tests
def test_gridspec_validation():
    '''
    Given: An invalid overlap specification.
    When: `GridSpec` is initialized.
    Then: Raise a ValueError in post-init.
    '''
    with pytest.raises(ValueError, match='Overlap must be smaller'):
        grid_layout.GridSpec(
            crs='EPSG:32617',
            origin=(500000.0, 5000000.0),
            pixel_size=(10.0, 10.0),
            tile_size=(256, 256),
            tile_stride=(300, 128),
            grid_extent=(5120.0, 5120.0),
        )


# ----- `GridLayout` tests
def test_gridlayout_generation():
    '''
    Given: A GridSpec with spatial extent.
    When: `GridLayout` is constructed.
    Then: Correct row/col tiles are generated within extent boundary
        constraints.
    '''
    spec = grid_layout.GridSpec(
        crs='EPSG:32617',
        origin=(0.0, 1000.0),
        pixel_size=(10.0, 10.0),
        tile_size=(256, 256),
        tile_stride=(128, 128),
        grid_extent=(5120.0, 5120.0), # 512x512 pixels
    )

    layout = grid_layout.GridLayout(spec)

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


def test_gridlayout_container_protocol():
    '''
    Given: A constructed `GridLayout`.
    When: Running dictionary protocol checks.
    Then: Iteration, length, keys, index validation, and value type
        checks succeed.
    '''
    spec = grid_layout.GridSpec(
        crs='EPSG:32617',
        origin=(0.0, 1000.0),
        pixel_size=(10.0, 10.0),
        tile_size=(256, 256),
        tile_stride=(128, 128),
        grid_extent=(1280.0, 1280.0),
    )
    layout = grid_layout.GridLayout(spec)

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
    spec = grid_layout.GridSpec(
        crs='EPSG:32617',
        origin=(500000.0, 5000000.0),
        pixel_size=(10.0, 10.0),
        tile_size=(256, 256),
        tile_stride=(128, 128),
        grid_extent=(2560.0, 2560.0),
    )
    layout = grid_layout.GridLayout(spec)

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
    spec = grid_layout.GridSpec(
        crs='EPSG:32617',
        origin=(0.0, 1000.0),
        pixel_size=(10.0, 10.0),
        tile_size=(256, 256),
        tile_stride=(128, 128),
        grid_extent=(3840.0, 3840.0),
    )
    layout = grid_layout.GridLayout(spec)

    payload = layout.to_payload()
    restored = grid_layout.GridLayout.from_payload(payload)

    assert restored.extent == layout.extent
    assert len(restored) == len(layout)
    assert restored[(0, 0)] == layout[(0, 0)]


def test_gridlayout_identity_properties():
    '''
    Given: Two GridLayouts with identical or differing parameters.
    When: Evaluating affine_identity and block_identity properties.
    Then: Invariant to extent/stride, sensitive to origin/crs/pixel/tile.
    '''
    spec1 = grid_layout.GridSpec(
        crs='EPSG:32617',
        origin=(0.0, 1000.0),
        pixel_size=(10.0, 10.0),
        tile_size=(256, 256),
        tile_stride=(128, 128),
        grid_extent=(3840.0, 3840.0),
    )
    spec2 = grid_layout.GridSpec(
        crs='EPSG:32617',
        origin=(0.0, 1000.0),
        pixel_size=(10.0, 10.0),
        tile_size=(256, 256),
        tile_stride=(64, 64),
        grid_extent=(5120.0, 5120.0),
    )
    spec3 = grid_layout.GridSpec(
        crs='EPSG:32617',
        origin=(0.0, 1000.0),
        pixel_size=(10.0, 10.0),
        tile_size=(512, 512),
        tile_stride=(128, 128),
        grid_extent=(3840.0, 3840.0),
    )
    layout1 = grid_layout.GridLayout(spec1)
    layout2 = grid_layout.GridLayout(spec2)
    layout3 = grid_layout.GridLayout(spec3)

    # affine_identity is invariant to extent, stride, and tile_size
    assert layout1.affine_identity == layout2.affine_identity
    assert layout1.affine_identity == layout3.affine_identity
    assert layout1.affine_identity == 'EPSG:32617|(0.0, 1000.0)|(10.0, -10.0)'

    # block_identity includes tile_size, but is invariant to stride/extent
    assert layout1.block_identity == layout2.block_identity
    assert layout1.block_identity != layout3.block_identity
    assert layout1.block_identity == (
        'EPSG:32617|(0.0, 1000.0)|(10.0, -10.0)|(256, 256)'
    )


# ----- grid persistence and report helpers tests
def test_gridlayout_from_fpath(tmp_path):
    '''
    Given: A persisted GridLayout artifact on disk.
    When: Loaded via `GridLayout.from_fpath` and `load_grid_from_fpath`.
    Then: Both return reconstructed GridLayout instances matching.
    '''
    spec = grid_layout.GridSpec(
        crs='EPSG:32617',
        origin=(0.0, 1000.0),
        pixel_size=(10.0, 10.0),
        tile_size=(256, 256),
        tile_stride=(128, 128),
        grid_extent=(3840.0, 3840.0),
    )
    layout = grid_layout.GridLayout(spec)
    grid_fp = str(tmp_path / 'grid.json')
    ctrl = artifacts.PayloadController[
        list[list[int]], grid_layout.GridMeta
    ](
        grid_fp,
        schema_id=grid_layout.GridLayout.SCHEMA_ID,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
    )
    ctrl.save(layout.to_payload())

    from_cls = grid_layout.GridLayout.from_fpath(grid_fp)
    assert from_cls.gid == layout.gid
    assert len(from_cls) == len(layout)

    from_func = grid_layout.load_grid_from_fpath(grid_fp)
    assert from_func.gid == layout.gid


def test_grid_report_helpers(tmp_path):
    '''
    Given: An output directory and a serialized grid report JSON.
    When: `get_grid_report_fpath` and `read_grid_report` are called.
    Then: Resolve report path and extract WorldGridReport payload.
    '''
    expected_fp = os.path.join(str(tmp_path), 'grid_report.json')
    assert grid_layout.get_grid_report_fpath(str(tmp_path)) == expected_fp

    grid_report: contracts.WorldGridReport = {
        'grid_fpath': '/path/to/grid.json',
        'grid_id': 'ontario_grid',
        'crs': 'EPSG:3161',
        'pixel_size': (10.0, 10.0),
        'tile_size': (512, 512),
        'tile_overlap': (64, 64),
    }
    report_data: contracts.GridReportSchema = {
        'run_id': 'world-grid',
        'timestamp': '2026-09-18T00:00:00',
        'status': 'SUCCESS',
        'grid': grid_report,
        'total_tiles': 42,
    }
    artifacts.Controller[contracts.GridReportSchema](
        expected_fp
    ).persist(report_data)

    loaded = grid_layout.read_grid_report(expected_fp)
    assert loaded['grid_id'] == 'ontario_grid'
    assert loaded['grid_fpath'] == '/path/to/grid.json'
