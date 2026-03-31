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

'''
I/O utilites for data_blocks.
'''

# standard imports
import os
# local imports
import landseg.geopipe.foundation.data_blocks as data_blocks
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def save_mapped_windows(
    grid_id: str,
    mapped_windows: data_blocks.MappedRasterWindows,
    dirpath: str
) -> None:
    '''doc'''

    # prepare output dir
    os.makedirs(dirpath, exist_ok=True)

    save_path = f'{dirpath}/windows_{grid_id}.pkl'
    utils.write_pickle(save_path, mapped_windows)
    utils.hash_artifacts(save_path)

def load_mapped_windows(
    grid_id: str,
    dirpath: str
) -> data_blocks.MappedRasterWindows:
    '''doc'''

    load_path = f'{dirpath}/windows_{grid_id}.pkl'
    windows: data_blocks.MappedRasterWindows = utils.load_pickle(load_path)
    return windows
