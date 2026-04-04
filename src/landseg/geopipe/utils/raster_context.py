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
Context manager utility for rasterio.open(...).
'''

# standard imports
import contextlib
import os
import typing
# third-party imports
import rasterio
import rasterio.io

@contextlib.contextmanager
def open_rasters(
        *rasters: str | None
    ) -> typing.Iterator[tuple[rasterio.io.DatasetReader | None, ...]]:
    '''
    Open multiple rasters safely and yield a tuple of `DatasetReader`.

    Accepts any number of filepaths (or None). Existing paths are opened
    via rasterio, None values are preserved, and all files are closed
    automatically on exit.
    '''

    with contextlib.ExitStack() as stack:
        opened_rasters: list[rasterio.io.DatasetReader | None] = []

        for raster in rasters:
            if isinstance(raster, str):
                assert os.path.exists(raster), f'Raster not found: {raster}'
                opened_raster = stack.enter_context(rasterio.open(raster))
                opened_rasters.append(opened_raster)
            else:
                opened_rasters.append(None)

        yield tuple(opened_rasters)