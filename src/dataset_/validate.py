'''doc'''

# standard imoprts
import os
import typing
# local imports
import utils

def validate_schema(
    world_grid_id: str,
    cache_root: str,
    logger: utils.Logger
) -> int:
    '''doc'''

    # initial status
    status = 0

    # expected schema path
    schema_path = f'{cache_root}/schema.json'
    # early exit if schema not found
    if not os.path.exists(schema_path):
        return 2

    # load schema into dict
    schema: dict[str, typing.Any] = utils.load_json(schema_path)
    # checks
    # dataset name match
    if os.path.basename(cache_root) != schema['dataset']['name']:
        logger.log('ERROR', 'Dataset name not matching')
        status = 1
    # world grid match
    if world_grid_id != schema['dataset']['world_grid']:
        logger.log('ERROR', 'World grid not matching')
        status = 1
    # fit data artifact integrity
    if not _check_hash(schema['normalization']['fit_stats_file']):
        logger.log('ERROR', 'Fit data stats missing/corrupted')
        status = 1
    if not _check_hash(schema['splits']['train_blocks']):
        logger.log('ERROR', 'Train blocks index missing/corrupted')
        status = 1
    if not _check_hash(schema['splits']['val_blocks']):
        logger.log('ERROR', 'Val blocks index missing/corrupted')
        status = 1
    if not _check_hash(schema['training_stats']['class_counts_train']):
        logger.log('ERROR', 'Train class count missing/corrupted')
        status = 1
    if not _check_hash(schema['training_stats']['class_counts_global']):
        logger.log('ERROR', 'Global class count missing/corrupted')
        status = 1
    # test data artifact integrity (if provided)
    if schema['dataset']['has_test_data']:
        if not _check_hash(schema['normalization']['test_stats_file']):
            logger.log('ERROR', 'Test data stats missing/corrupted')
            status = 1
        if not _check_hash(schema['splits']['val_blocks']):
            logger.log('ERROR', 'Test blocks index missing/corrupted')
            status = 1

    # all passed
    return status

def _check_hash(obj: dict[str, str]) -> bool:
    '''doc'''

    fpath = obj.get('fpath', '')
    hash_rec = obj.get('sha256', '')
    if not (os.path.exists(fpath) and hash_rec):
        return False
    hash_value = utils.hash_artifacts(fpath, write_to_record=False)
    if hash_value != hash_rec:
        return False
    return True
