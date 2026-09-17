# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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
Dataset manifest compilation and aggregation utilities.

This module loads root dataset manifest JSON files, validates linked
manifest entries, and compiles them into a path-indexed mapping of
normalized `ManifestEntry` objects.

Public APIs:
    - `DatasetManifestError`: Error raised during manifest compilation.
    - `compile_dataset_manifest`: Read and validate dataset manifest JSON.
'''

# standard imports
from __future__ import annotations
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.harmonize.manifest as manifest


# ----- typing aliases
ManifestCtrl = artifacts.Controller[list[dict[str, typing.Any]]]
ManifestEntryCtrl = artifacts.Controller[manifest.ManifestEntry]


# ----- public classes
class DatasetManifestError(Exception):
    '''Base class for errors when compiling dataset manifest JSON.'''

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


# ----- public functions
def compile_dataset_manifest(fp: str) -> dict[str, manifest.ManifestEntry]:
    '''
    Read, validate, and compile dataset manifest from JSON.

    Args:
        fp:
            File path to root dataset manifest JSON.

    Returns:
        dict[str, manifest.ManifestEntry]:
            Mapping of raster file paths to normalized manifest entries.
    '''
    # load JSON via artifact controller
    ctrl = ManifestCtrl.load_json_or_fail(fp)
    ctrl.hash(overwrite=False) # hash once
    mfst = ctrl.fetch()

    # expect JSON read as a list of dicts
    if not isinstance(mfst, list):
        raise DatasetManifestError(
            f'Manifest JSON expected to read as a list dictionaries, '
            f'got: {type(mfst)}'
        )

    compiled: list[manifest.ManifestEntry] = []
    for i, mfst_entry in enumerate(mfst):
        try:
            mfst_entry_path = mfst_entry.get('manifest', '')
            _ctrl = ManifestEntryCtrl.load_json_or_fail(mfst_entry_path)
            _ctrl.hash(overwrite=False) # hash once
            entry = _ctrl.fetch()
        except (TypeError, ValueError) as e:
            raise DatasetManifestError(f'Invalid manifest entry {i}') from e

        try:
            norm = manifest.ManifestEntryNormalizer(entry).normalized_entry
            compiled.append(norm)
        except(TypeError, ValueError) as e:
            raise DatasetManifestError(f'Invalid manifest entry {i}') from e

        # matching sanity
        assert norm['name'] == mfst_entry.get('name')
        assert norm['path'] == mfst_entry.get('path')

    # return a dict indexed by file path
    return {c['path']: c for c in compiled}
