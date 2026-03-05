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

'''Simple top-level namesapce for `landseg.utils`.'''

from .cfg_access import ConfigAccess
from .contxt import open_rasters
from .funcs import(
    get_dir_size,
    get_timestamp,
    hash_artifacts,
    hash_payload,
    load_json,
    load_pickle,
    write_json,
    write_pickle,
)
from .logger import Logger
from .multip import ParallelExecutor
from .pca import pca_transform
from .preview import export_previews

__all__ = [
    # classes
    'ConfigAccess',
    'Logger',
    'ParallelExecutor',
    # functions
    'export_previews',
    'get_dir_size',
    'get_timestamp',
    'hash_artifacts',
    'hash_payload',
    'load_json',
    'load_pickle',
    'open_rasters',
    'pca_transform',
    'write_json',
    'write_pickle',
    # types
]
