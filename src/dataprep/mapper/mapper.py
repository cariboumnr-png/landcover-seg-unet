'''doc'''

# standard imports
import copy
import dataclasses
# third-party imports
import rasterio
# local imports
import alias
import dataprep
import grid
import utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class DataWindows:
    '''Container for input read windows and expected window shape.'''
    image_windows: alias.RasterWindowDict   # indexed read windows
    label_windows: alias.RasterWindowDict   # indexed read windows can be empty
    expected_shape: tuple[int, int]  # expected window shape (W*H) in px

def map_rasters(
    world_grid: grid.GridLayout,
    mode: str,
    config: dataprep.InputConfig,
    logger: utils.Logger
) -> DataWindows:
    '''doc'''

    # mode selection
    if mode == 'fit':
        image_fpath=config['fit_input_img']
        label_fpath=config['fit_input_lbl']
    elif mode == 'test':
        assert config['test_input_img'] # sanity type check
        image_fpath=config['test_input_img']
        label_fpath=None
    else:
        raise ValueError(f'Invalid builder mode {mode}')

    # get geometry summary
    geom = dataprep.validate_geometry(image_fpath, label_fpath, logger)

    # alignment to the world grid
    # check CRS match
    grid_crs = world_grid.crs
    data_crs = geom['crs']
    if grid_crs != data_crs:
        raise ValueError(f'Data CRS [{data_crs}] != world CRS [{grid_crs}]')

    # focus on tiles inside the intersected extent
    inside = _crop(world_grid, geom)

    # get image and optionally label reading windows
    img_windows = _get_windows(world_grid, geom['image_transform'], inside)
    lbl_windows = _get_windows(world_grid, geom['label_transform'], inside)

    # return
    return DataWindows(img_windows, lbl_windows, world_grid.tile_size)

def _crop(
    world_grid: grid.GridLayout,
    geom_summary: dataprep.GeometrySummary,
) -> list[tuple[int, int]]:
    '''doc'''

    # prep return list
    inside: list[tuple[int, int]] = []
    # get world origin
    gx, gy = world_grid.origin
    # get pixel size and bbox
    px, py = geom_summary['pixel_size'] # in CRS units per pixel
    l, b, r, t = geom_summary['inter_bbox'] # in CRS units
    # get tile size to CRS units
    tile_x = world_grid.tile_size[1] * px
    tile_y = world_grid.tile_size[0] * py
    # iterate through world grid
    for (x, y) in world_grid.keys(): # keys (x, y) are in pixels
        # unify comparisons to be between CRS coordinates
        if (
            gx + x * px < r and             # right side of tile < R
            gy - y * py > b and             # bottom side of tile > B
            gx + x * px > l - tile_x and    # left side of tile > L
            gy - y * py < t + tile_y        # top side of tile < T
        ):
            inside.append((x, y))
    return inside

def _get_windows(
    world_grid: grid.GridLayout,
    transform: rasterio.Affine | None,
    inside_idx: list[tuple[int, int]]
) -> dict[tuple[int, int], alias.RasterWindow]:
    '''doc'''

    # get raster reading windows - empty when transform is None
    windows: dict[tuple[int, int], alias.RasterWindow] = {}
    if transform is not None:
        # set grid offset for label
        _grid = copy.deepcopy(world_grid)
        _grid.offset_from(transform)
        # get windows only inside
        for idx, window in _grid.items():
            if idx in inside_idx:
                windows[idx] = window
    # return
    return windows
