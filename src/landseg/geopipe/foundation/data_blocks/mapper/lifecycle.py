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
import landseg.geopipe.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.data_blocks.mapper as mapper
import landseg.utils as utils

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
    img, lbl = image_label_fpaths

    # mapped windows fpath
    logger.log('INFO', f'Try to load mapped windows from {gid}')
    windows_fpath = f'{artifacts_dir}/windows_{gid}.pkl'
    mapped_windows: mapper.MappedRasterWindows
    load_status, m, mapped_windows = artifacts.load_pickle_hash(windows_fpath)
    if load_status: # non-zero status indicates false artifact -> rebuild
        logger.log('INFO', f'Mapped windows loading error: {m}')
        build = True
    else:
        logger.log('INFO', f'Mapped windows from {gid} loaded')
        build = False

    # policy: build if missing
    if policy is artifacts.LifecyclePolicy.BUILD_IF_MISSING:
        pass
    # policy: force rebuild
    elif policy is artifacts.LifecyclePolicy.REBUILD:
        build = True
    # unsupported policy
    else:
        msg = f'Currently unsupported policy: {policy}'
        logger.log('ERROR', msg)
        raise NotImplementedError(msg)

    # build if needed
    if build:
        mapped_windows = mapper.map_rasters(world_grid, img, lbl, logger)
        artifacts.write_pickle_hash(windows_fpath, mapped_windows)
        logger.log('INFO', f'Mapped windows from {gid} created')

    # return
    return mapped_windows
