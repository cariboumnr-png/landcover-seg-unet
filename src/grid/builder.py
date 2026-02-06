'''
Docstring for grid.builder
'''

# standard imports
import functools
import os
# third-party import
import rasterio
# local imports
import _types
import grid
import utils

# -------------------------------Public Function-------------------------------
def build_grid(
    mode: str,
    base_config: _types.ConfigType,
    extent_mode: _types.ConfigType
):
    '''doc'''

    # get gridspec from configs
    spec = _get_gridspec(base_config, extent_mode)

    # build grid by mode
    output_grid = grid.GridLayout(mode, spec)

    # return
    return output_grid

# ------------------------------private  function------------------------------
def _get_gridspec(
    base_config: _types.ConfigType,
    extent_mode: _types.ConfigType
) -> grid.GridSpec:
    '''doc'''

    # config accessor
    base = utils.ConfigAccess(base_config)
    mode = utils.ConfigAccess(extent_mode)

    # persistent config from base
    partial_gridspec = functools.partial(
        grid.GridSpec,
        crs=base.get_option('crs'),
        tile_size=(
            base.get_option('tile_size', 'row'),
            base.get_option('tile_size', 'col')
        ),
        tile_overlap=(
            base.get_option('tile_overlap', 'row'),
            base.get_option('tile_overlap', 'col')
        )
    )

    # get extent based on mode
    # from reference raster (auto)
    if mode.get_option('mode') == 'ref':
        ref_raster = mode.get_option('ref_raster_fpath')
        if not os.path.exists(ref_raster):
            raise ValueError('Reference raster not provided')
        # open reference raster
        with rasterio.open(ref_raster) as src:
            # get transform - pixel size
            transform = src.transform
            px, py = transform.a, abs(transform.e)
            # get bounding box - origin and extent
            l, b, r, t = src.bounds
            # assign to gridspec
            return partial_gridspec(
                pixel_size=(px, py),        # pixel size in x, y
                origin=(l, t),              # left, ,top as x, y
                grid_extent=(t - b, r - l)  # top-bottom as H, right-left as W
            )
    # from aoi (manual)
    elif mode.get_option('mode') == 'aoi':
        return partial_gridspec(
            pixel_size=(
                mode.get_option('pixel_size', 'x'),
                mode.get_option('pixel_size', 'y')
            ),
            origin=(
                mode.get_option('origin', 'x'),
                mode.get_option('origin', 'y')
            ),
            grid_extent=(
                mode.get_option('grid_extent', 'hy'),
                mode.get_option('grid_extent', 'wx'),
            )
        )
    # from tiles count (manual)
    elif mode.get_option('mode') == 'tiles':
        return partial_gridspec(
            pixel_size=(
                mode.get_option('pixel_size', 'x'),
                mode.get_option('pixel_size', 'y')
            ),
            origin=(
                mode.get_option('origin', 'x'),
                mode.get_option('origin', 'y')
            ),
            grid_shape=(
                mode.get_option('grid_shape', 'row'),
                mode.get_option('grid_shape', 'col'),
            )
        )
    #
    raise ValueError(f'Invalid extent mode: {mode.get_option('mode')}')
