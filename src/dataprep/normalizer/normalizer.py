'''Data blocks preperation pipeline.'''

# third-party imports
import numpy
# local imports
import dataprep
import utils

# -------------------------------Public Function-------------------------------
def normalize_data_blocks(
    input_type: str,
    config: dataprep.OutputConfig,
    logger: utils.Logger,
    *,
    renormalize = False
) -> None:
    '''doc'''

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
    ss = dataprep.get_image_stats(fit_blks, stats_fpath, logger)
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

    data = dataprep.DataBlock().load(block_fpath).data
    if data.image_normalized.size != data.image.size or \
        numpy.isnan(data.image_normalized).any():
        return {'invalid': block_fpath}
    return {'passed': block_fpath}

def _normalize_block(
        fpath: str,
        stats: dict,
    ) -> None:
    '''doc.'''

    rb = dataprep.DataBlock().load(fpath)
    mmin, mmax = rb.normalize_image(stats)
    assert mmin > -100 and mmax < 100
    rb.save(fpath)
