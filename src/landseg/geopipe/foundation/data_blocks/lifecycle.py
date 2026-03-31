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

# standard imports
from __future__ import annotations
# local imports
import landseg.geopipe.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.data_blocks as data_blocks
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def prepare_mapped_raster_windows(
    world_grid: geo_core.GridLayout,
    image_label_fpaths: tuple[str, str],
    logger: utils.Logger,
    *,
    artifacts_dir: str,
    policy: artifacts.LifecyclePolicy
) -> data_blocks.MappedRasterWindows:
    '''doc'''

    # aliases
    gid = world_grid.gid
    img, lbl = image_label_fpaths

    if policy is artifacts.LifecyclePolicy.BUILD_IF_MISSING:
        logger.log('INFO', f'Try to load mapped windows from {gid}')
        try:
            windows = data_blocks.load_mapped_windows(gid, artifacts_dir)
            logger.log('INFO', f'Mapped windows from {gid} loaded')
            return windows
        except FileNotFoundError:
            logger.log('INFO', f'Mapped windows from {gid} not found')
            windows = data_blocks.map_rasters(world_grid, img, lbl, logger)
            data_blocks.save_mapped_windows(gid, windows, artifacts_dir)
            logger.log('INFO', f'Mapped windows from {gid} created')
            return windows

    # unsupported policy
    else:
        msg = f'Currently unsupported policy: {policy}'
        logger.log('ERROR', msg)
        raise NotImplementedError(msg)
