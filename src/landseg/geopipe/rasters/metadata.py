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

'''
Raster metadata and GDAL tag inspection utilities.

This module provides functions to inspect band descriptions, decode
embedded VRT tags, and aggregate composite band metadata tags.

Public APIs:
    - `parse_vrt_tag`: Decode serialized JSON or literal VRT tag value.
    - `read_band_map`: Return lower-case band-description to index mapping.
    - `read_composite_band_tags`: Collect composite per-band metadata tags.
'''

# standard imports
import ast
import json
import pathlib
import typing
# local imports
import landseg.geopipe.rasters.io as io


# ----- public functions
def parse_vrt_tag(value: str) -> typing.Any:
    '''Decode GDAL metadata serialized as JSON or a Python literal.'''
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value


def read_band_map(fpath: pathlib.Path) -> dict[str, int]:
    '''
    Return lower-case band-description to zero-based index mapping.

    Args:
        fpath:
            File path to the raster dataset.

    Returns:
        dict[str, int]:
            Mapping from lowercase band description to zero-based index.
    '''
    descriptions = io.get_info(fpath).descriptions
    names: list[str] = []

    for index, description in enumerate(descriptions):
        if description and description.strip():
            name = description.strip().lower()
        else:
            name = f'band_{index + 1}'
        names.append(name)

    if len(set(names)) != len(names):
        return {}

    return {name: index for index, name in enumerate(names)}


def read_composite_band_tags(paths: list[pathlib.Path]) -> list[dict[str, str]]:
    '''
    Collect composite per-band metadata tags across source rasters.

    Args:
        paths:
            List of file paths for source rasters.

    Returns:
        list[dict[str, str]]:
            Sequential band metadata dictionaries for stacked VRT.
    '''
    metadata: list[dict[str, str]] = []
    for source_path in paths:
        info = io.get_info(source_path)
        dataset_tags = info.tags
        for band_tags in info.band_tags:
            tags = {**dataset_tags, **band_tags}
            metadata.append(tags)
    return metadata
