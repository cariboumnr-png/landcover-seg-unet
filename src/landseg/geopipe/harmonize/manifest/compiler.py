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
import collections
import pathlib
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.harmonize.manifest.normalizer as normalizer
import landseg.utils.validator as val


# ----- public classes
class DatasetManifestError(Exception):
    '''Base class for errors when compiling dataset manifest JSON.'''

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


# ----- public functions
def compile_dataset_manifest(
    fp: pathlib.Path
) -> tuple[normalizer.ManifestEntry, ...]:
    '''
    Read, validate, and compile dataset manifest from JSON.

    Args:
        fp:
            File path to root dataset manifest JSON.

    Returns:
        tuple[schema.ManifestEntry, ...]:
            Tuple of normalized manifest entries.
    '''
    # load JSON via artifact controller
    ctrl = artifacts.Controller[list[dict[str, typing.Any]]](fp)
    ctrl.hash(overwrite=False) # hash once
    mfst = ctrl.fetch()
    # cheap type check: expect JSON read as a list of dicts
    try:
        val.require_sequence(mfst, 'dataset_manifest')
        if len(mfst) == 0:
            raise DatasetManifestError('Manifest JSON read as an empty list')
        for i, mfst_entry in enumerate(mfst):
            val.require_mapping(mfst_entry, f'manifest_entry_{i}')
    except val.ValidationError as e:
        raise DatasetManifestError(
            f'Manifest JSON expected to read as a list dictionaries, '
            f'got: {type(mfst)}'
        ) from e

    compiled: list[normalizer.ManifestEntry] = []
    fingerprints: dict[str, str] = {}
    for mfst_entry in mfst:
        name = val.require_string(mfst_entry.get('name'), 'entry_name')
        data_path = val.require_file(mfst_entry.get('path'), 'data_file_path')
        mfst_path = val.require_file(mfst_entry.get('manifest'), 'manifest')

        try:
            _ctrl = artifacts.Controller[normalizer.ManifestEntry](mfst_path)
            _ctrl.hash(overwrite=False) # hash once
            entry = _ctrl.fetch()
        except artifacts.ArtifactError as e:
            raise DatasetManifestError(f'Error loading {mfst_path}') from e

        try:
            norm = normalizer.normalize_manifest_entry(entry)
            compiled.append(norm)
        except (ValueError, val.ValidationError) as e:
            raise DatasetManifestError(f'Error validating {mfst_path}') from e

        # matching checks
        if not (
            name == norm['name'] and
            data_path.resolve() == norm['path'].resolve()
        ):
            raise DatasetManifestError('Mismatch manifest name/data file path')

        fingerprints[name] = artifacts.compute_fingerprint(norm)

    # check for collisions
    dup = {i for i, c in collections.Counter(fingerprints).items() if c > 1}
    if dup:
        raise DatasetManifestError(f'Found duplicated manifest entries: {dup}')

    # return a tuple of entries
    return tuple(compiled)
