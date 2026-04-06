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

'''Data blocks artifacts lifecycle management.'''

# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.common.alias as alias
import landseg.geopipe.foundation.data_blocks.mapper as mapper
import landseg.utils as utils

# typing aliases
MappingCtrl = artifacts.Controller[dict]

# -------------------------------Public Function-------------------------------
def map_rasters_to_grid(
    world_grid: geo_core.GridLayout,
    image_label_fpaths: tuple[str, str],
    logger: utils.Logger,
    *,
    artifacts_dir: str,
    policy: artifacts.LifecyclePolicy
) -> mapper.MappedRasterWindows:
    '''doc'''

    # aliases
    gid = world_grid.gid
    image, label = image_label_fpaths

    # artifacts controller
    ctrl = MappingCtrl(f'{artifacts_dir}/windows_{gid}.json', 'json', policy)

    # mapped windows fpath
    logger.log('INFO', f'Try to load mapped windows from {gid}')
    payload = ctrl.fetch()

    # build if needed
    if not payload:
        mapped_windows = mapper.map_rasters(world_grid, image, label, logger)
        payload = {
            'grid_id': mapped_windows.grid_id,
            'tile_shape': list(mapped_windows.tile_shape),
            'image': _canonicalize(mapped_windows.image),
            'label': _canonicalize(mapped_windows.label)
        }
        ctrl.persist(payload)
        logger.log('INFO', f'Mapped windows from {gid} created')

    else:
        mapped_windows = mapper.MappedRasterWindows(
            grid_id=payload['grid_id'],
            tile_shape=tuple(payload['tile_shape']),
            image=_parse(payload['image']),
            label=_parse(payload['label'])
        )

    # return
    return mapped_windows

def _canonicalize(mapped_windows: alias.RasterWindowDict) -> list[list[int]]:
    '''Create a canonical serialization for mapped windows.'''

    canon: list[list[int]] = []
    for k, w in sorted(mapped_windows.items()):
        canon.append(
            [k[0], k[1], w.col_off, w.row_off, w.width, w.height]
        )
    return canon

def _parse(payload: list[list[int]]) -> alias.RasterWindowDict:
    '''Parse window dict from payload.'''

    parsed: alias.RasterWindowDict = {}
    for c in payload:
        x, y, col_off, row_off, w, h = c
        window = alias.RasterWindow(col_off, row_off, w, h) # type: ignore
        parsed[(x, y)] = window
    return parsed
