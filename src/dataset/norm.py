'''Data blocks preperation pipeline.'''

# standard imports
import math
import os
import random
# third-party imports
import numpy
import tqdm
# local imports
import alias
import dataset
import utils

# -------------------------------Public Function-------------------------------
def normalize_dataset(
        dataset_name: str,
        cache_config: alias.ConfigType,
        logger: utils.Logger,
        mode: str
    ):
    '''doc'''

    # get a child logger
    logger=logger.get_child('stats')

    # config accessors
    cache_cfg = utils.ConfigAccess(cache_config)

    # get artifacts names
    lbl_count = cache_cfg.get_asset('artifacts', 'label', 'count_training')
    if mode == 'training':
        work_blks = cache_cfg.get_asset('artifacts', 'blocks', 'valid')
    elif mode == 'inference':
        work_blks = cache_cfg.get_asset('artifacts', 'blocks', 'square')
    else:
        raise ValueError('Mode must be either "training" or "inference".')
    image_stats = cache_cfg.get_asset('artifacts', 'image', 'stats')

    # training or inference blocks dirpath and artifact filepaths
    _dir = f'./data/{dataset_name}/cache/{mode}'
    lbl_count_fpath = os.path.join(_dir, lbl_count)
    blks_fpath = os.path.join(_dir, work_blks)
    stats_fpath = os.path.join(_dir, image_stats)

    # count label classes only on training blocks
    if mode == 'training':
        dataset.count_label_cls(
            blks_fpaths=blks_fpath,
            results_fpath=lbl_count_fpath,
            logger=logger,
            overwrite=cache_cfg.get_option('flags', 'overwrite_counts')
        )

    # get image stats on blocks
    _get_image_stats(
        blks_fpaths=blks_fpath,
        stats_fpath=stats_fpath,
        logger=logger,
        overwrite=cache_cfg.get_option('flags', 'overwrite_stats')
    )

    # normalize blocks
    _normalize_blocks(
        blks_fpaths=blks_fpath,
        stats_fpath=stats_fpath,
        logger=logger,
        overwrite=cache_cfg.get_option('flags', 'overwrite_stats')
    )

# ------------------------------private  function------------------------------
# global image stats aggregated from selected blocks
def _get_image_stats(
        blks_fpaths: str,
        stats_fpath: str,
        logger: utils.Logger,
        *,
        overwrite: bool
    ) -> dict[str, dict[str, int | float]]:
    '''Aggregate stats from a given list of blocks'''

    # check if stats are complete
    logger.log('INFO', 'Checking block image stats')
    valid_blks: dict[str, str] = utils.load_json(blks_fpaths)
    stats_complete = _validate_image_stats(list(valid_blks.values()), logger)

    # load stats if file already exists
    if os.path.exists(stats_fpath) and stats_complete and not overwrite:
        logger.log('INFO', f'Gathering image stats from: {stats_fpath}')
        return utils.load_json(stats_fpath)

    logger.log('INFO', 'Aggregating global image stats')
    # read a random block to get number of image channels
    temp = dataset.DataBlock()
    temp.load(random.choice(list(valid_blks.values())))
    num_bands = len(temp.meta['block_image_stats'])

    # define a return dict
    stats_dict = {
        f'band_{_}': {
            'total_count': 0,
            'current_mean': 0.0,
            'accum_m2': 0.0,
            'std': 0.0
        } for _ in range(0, num_bands)
    }

    # iterate through provided block files
    for fpath in tqdm.tqdm(valid_blks.values()):
        # prep
        rb = dataset.DataBlock().load(fpath)
        stats = rb.meta['block_image_stats']
        # return dict and stats dict have the same keys
        for key, value_dict in stats.items():
            stats_dict[key] = _welfords_online(value_dict, stats_dict[key])

    # deviation to std
    for v in stats_dict.values():
        v['std'] = math.sqrt((v['accum_m2'] / (v['total_count'] - 1)))

    # save results to file and return
    logger.log('INFO', f'Global image stats save to {stats_fpath}')
    utils.write_json(stats_fpath, stats_dict)
    return stats_dict

def _validate_image_stats(
        blks_fpaths: list[str],
        logger: utils.Logger,
    ) -> bool:
    '''doc'''

    jobs = [(_check_block_image_stats, (f, ), {}) for f in blks_fpaths]
    rr: list[dict] = utils.ParallelExecutor().run(jobs)
    work_fpaths = [r.get('restats', 0) for r in rr if r.get('restats', 0)]
    # fix if any blocks have invalida stats
    if not work_fpaths:
        logger.log('INFO', 'All blocks have complete stats')
        return True
    logger.log('INFO', f'Found {len(work_fpaths)} blocks with bad stats')
    for fpath in tqdm.tqdm(work_fpaths):
        rb = dataset.DataBlock().load(fpath)
        rb.recalculate_stats(fpath) # save to overwrite
    logger.log('INFO', f'Updated stats for {len(work_fpaths)} blocks')
    return False

def _check_block_image_stats(block_fpath: str) -> dict[str, str]:
    '''doc'''

    meta = dataset.DataBlock().load(block_fpath).meta
    stats = meta['block_image_stats']
    for value_dict in stats.values():
        if any(numpy.isnan(x) for x in value_dict.values()):
            return {'restats': block_fpath}
    return {'passed': block_fpath}

def _welfords_online(
        input_stats: dict[str, int | float],
        current_results: dict[str, int | float]
    ) -> dict[str, int | float]:
    '''Welford's Online Algorithm.'''

    # block stats from stats dict
    nb = input_stats['count']
    mb = input_stats['mean']
    m2b = input_stats['m2']
    # current stats from the processed blocks
    nt = current_results['total_count']
    mt = current_results['current_mean']
    dt = current_results['accum_m2']
    # Welford's online algorithm
    delta = mb - mt
    mt += delta * (nb / (nt + nb))
    dt += m2b + (delta ** 2) * nt * nb / (nt + nb)
    nt += nb
    # assign back and return
    current_results['total_count'] = nt
    current_results['current_mean'] = mt
    current_results['accum_m2'] = dt
    return current_results

# normalize blocks using global image stats
def _normalize_blocks(
        blks_fpaths: str,
        stats_fpath: str,
        logger: utils.Logger,
        *,
        overwrite: bool
    ) -> None:
    '''Normalize blocks using provided global stats dict.'''

    blk_fpaths: dict[str, str] = utils.load_json(blks_fpaths)

    # get blocks that needs to be updated for image normalization
    if overwrite:
        work_fpaths = blk_fpaths.values()
    else:
        # multiprocessing check blocks on normalized image channels
        logger.log('INFO', 'Checking block image normalization')
        jobs = [(_check_block_normal, (f, ), {}) for f in blk_fpaths.values()]
        rs: list[dict] = utils.ParallelExecutor().run(jobs)
        work_fpaths = [r.get('renorm', 0) for r in rs if r.get('renorm', 0)]
        logger.log('INFO', f'{len(work_fpaths)} blocks with faulty normalization')
    # exist if none to be updated
    if not work_fpaths:
        logger.log('INFO', 'No blocks need image normalization updates')
        return

    # parallel processing blocks
    stats: dict[str, dict[str, float]] = utils.load_json(stats_fpath)
    logger.log('INFO', 'Updating/overwriting block image normalization')
    jobs = [(_normalize_block, (f, stats,), {}) for f in work_fpaths]
    _ = utils.ParallelExecutor().run(jobs)
    logger.log('INFO', 'Image normalization completed')

def _check_block_normal(block_fpath: str) -> dict[str, str]:
    '''Check completeness of normalized image channel of a block.'''

    data = dataset.DataBlock().load(block_fpath).data
    if data.image_normalized.size != data.image.size or \
        numpy.isnan(data.image_normalized).any():
        return {'renorm': block_fpath}
    return {'passed': block_fpath}

def _normalize_block(
        fpath: str,
        stats: dict,
    ) -> None:
    '''doc.'''

    rb = dataset.DataBlock().load(fpath)
    mmin, mmax = rb.normalize_image(stats)
    assert mmin > -100 and mmax < 100
    rb.save(fpath)
