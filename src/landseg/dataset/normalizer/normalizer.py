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
Block normalization utilities for dataset blocks using global image
statistics. Supports validation and (re)normalization of cached blocks.

Public APIs:
    - normalize_blocks: Validate and normalize blocks for fit/test data.
'''

# third-party imports
import numpy
# local imports
import landseg.dataset as dataset
import landseg.dataset.blockbuilder as blockbuilder
import landseg.dataset.normalizer as normalizer
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def normalize_blocks(
    input_type: str,
    config: dataset.OutputConfig,
    logger: utils.Logger,
    *,
    renormalize = False
) -> None:
    '''
    Validate and normalize image channels of cached blocks.

    Args:
        input_type: Dataset split to process, either "fit" or "test".
        config: Output configuration with block/artifact paths.
        logger: Logger for progress updates and diagnostics.
        renormalize: If True, force normalization for all blocks; else
            only fix blocks detected with missing/invalid normalization.

    Raises:
        ValueError: If input_type is not "fit" or "test".
    '''

    # retrieve artifact paths by input data type
    if input_type == 'fit':
        input_blocks = config['fit_valid_blks']
        stats_fpath = config['fit_img_stats']
    elif input_type == 'test':
        input_blocks = config['test_all_blks']
        stats_fpath = config['test_img_stats']
    else:
        raise ValueError(f'Invalid data type: {input_type}')

    # get image stats for fit blocks
    # load valid fit blocks
    fit_blks_dict: dict[str, str] = utils.load_json(input_blocks)
    fit_blks: list[str] = list(fit_blks_dict.values())
    ss = normalizer.get_image_stats(fit_blks, stats_fpath, logger)
    # normalize blocks
    _normalize_blocks(fit_blks, ss, logger, renorm=renormalize)

# ------------------------------private  function------------------------------
def _normalize_blocks(
    input_blocks: list[str],
    stats: dict[str, dict[str, float]],
    logger: utils.Logger,
    *,
    renorm: bool = False
) -> None:
    '''Normalize blocks using provided global stats dict.'''

    # get blocks that needs to be updated for image normalization
    if renorm:
        to_fix = input_blocks
    else:
        # multiprocessing check blocks on normalized image channels
        logger.log('INFO', 'Checking block image normalization')
        jobs = [(_check_block_normal, (f, ), {}) for f in input_blocks]
        rs: list[dict] = utils.ParallelExecutor().run(jobs)
        to_fix = [r.get('invalid', 0) for r in rs if r.get('invalid', 0)]
        logger.log('INFO', f'{len(to_fix)} blocks with faulty normalization')
    # exist if none to be updated
    if not to_fix:
        logger.log('INFO', 'No blocks need image normalization updates')
        return

    # parallel processing blocks
    logger.log('INFO', 'Updating/overwriting block image normalization')
    jobs = [(_normalize_block, (f, stats,), {}) for f in to_fix]
    _ = utils.ParallelExecutor().run(jobs)
    logger.log('INFO', 'Image normalization completed')

def _check_block_normal(block_fpath: str) -> dict[str, str]:
    '''Check completeness of normalized image channel of a block.'''

    data = blockbuilder.DataBlock().load(block_fpath).data
    if data.image_normalized.size != data.image.size or \
        numpy.isnan(data.image_normalized).any():
        return {'invalid': block_fpath}
    return {'passed': block_fpath}

def _normalize_block(
        fpath: str,
        stats: dict,
    ) -> None:
    '''Apply normalization to a single block and persist the update.'''

    rb = blockbuilder.DataBlock().load(fpath)
    mmin, mmax = rb.normalize_image(stats)
    assert mmin > -100 and mmax < 100
    rb.save(fpath)
