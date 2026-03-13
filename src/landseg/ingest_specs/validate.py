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
Schema validation utilities for dataprep caches. Verifies dataset name,
grid ID, and artifact integrity via recorded SHA-256 checksums.

Public APIs:
    - validate_schema: Validate a dataset schema.json and referenced
      artifacts.
'''

# standard imoprts
import os
# local imports
import landseg.core.ingest_protocols as ingest_protocols
import landseg.utils as utils

def validate_schema(
    world_grid_id: str,
    cache_root: str,
    logger: utils.Logger
) -> int:
    '''
    Validate a dataset schema and referenced artifacts at given root.

    Args:
        world_grid_id: Expected world grid identifier.
        cache_root: Root directory containing schema.json and artifacts.
        logger: Logger for error reporting during validation.

    Returns:
        int: Validation status code
            * 0 -> all checks passed,
            * 1 -> mismatch or corrupted/missing artifacts,
            * 2 -> schema.json not found.
    '''

    # initial status
    status = 0

    # expected schema path
    schema_path = f'{cache_root}/schema.json'
    # early exit if schema not found
    if not os.path.exists(schema_path):
        return 2

    # load schema into dict
    schema: ingest_protocols.SchemaFull = utils.load_json(schema_path)
    # checks
    # dataset name match
    if os.path.basename(cache_root) != schema['dataset']['name']:
        logger.log('ERROR', 'Dataset name not matching')
        status = 1
    # world grid match
    if world_grid_id != schema['world_grid']['gid']:
        logger.log('ERROR', 'World grid not matching')
        status = 1
    # fit data artifact integrity
    if not _check_hash(
        schema['normalization']['fit_stats_file'],
        schema['checksums']['fit_stats_file']
    ):
        logger.log('ERROR', 'Fit data stats missing/corrupted')
        status = 1
    if not _check_hash(
        schema['splits']['train_blocks'],
        schema['checksums']['train_blocks']
    ):
        logger.log('ERROR', 'Train blocks index missing/corrupted')
        status = 1
    if not _check_hash(
        schema['splits']['val_blocks'],
        schema['checksums']['val_blocks']
    ):
        logger.log('ERROR', 'Val blocks index missing/corrupted')
        status = 1
    if not _check_hash(
        schema['training_stats']['class_counts_train'],
        schema['checksums']['class_counts_train']
    ):
        logger.log('ERROR', 'Train class count missing/corrupted')
        status = 1
    if not _check_hash(
        schema['training_stats']['class_counts_global'],
        schema['checksums']['class_counts_global']
    ):
        logger.log('ERROR', 'Global class count missing/corrupted')
        status = 1
    # test data artifact integrity (if provided)
    if schema['dataset']['has_test_data']:
        if not _check_hash(
            schema['normalization']['test_stats_file'],
            schema['checksums']['test_stats_file']
        ):
            logger.log('ERROR', 'Test data stats missing/corrupted')
            status = 1
        if not _check_hash(
            schema['splits']['test_blocks'],
            schema['checksums']['test_blocks']
        ):
            logger.log('ERROR', 'Test blocks index missing/corrupted')
            status = 1

    # all passed
    return status

def _check_hash(fpath: str, checksum: str) -> bool:
    '''Return True if an artifact exists and matches recorded hash.'''

    if not (os.path.exists(fpath) and checksum):
        return False
    hash_value = utils.hash_artifacts(fpath, write_to_record=False)
    if hash_value != checksum:
        return False
    return True
