'''doc'''

# standard imports
import math
import os
import random
# third-party imports
import numpy
import tqdm
# local imports
import landseg.dataprep.blockbuilder as blockbuilder
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def get_image_stats(
    input_blocks: list[str],
    stats_fpath: str,
    logger: utils.Logger,
    *,
    recompute: bool = False
) -> dict[str, dict[str, int | float]]:
    '''Aggregate stats from a given list of blocks'''

    # load stats if file already exists
    if os.path.exists(stats_fpath) and not recompute:
        logger.log('INFO', f'Gathering image stats from: {stats_fpath}')
        # check if stats are complete
        logger.log('INFO', 'Checking block image stats')
        stats_complete = _validate_image_stats(input_blocks, logger)
        if stats_complete:
            return utils.load_json(stats_fpath)

    logger.log('INFO', 'Aggregating global image stats')
    # read a random block to get number of image channels
    temp_meta = blockbuilder.DataBlock().load(random.choice(input_blocks)).meta
    num_bands = len(temp_meta['block_image_stats'])

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
    for fpath in tqdm.tqdm(input_blocks):
        # prep
        rb = blockbuilder.DataBlock().load(fpath)
        stats = rb.meta['block_image_stats']
        # return dict and stats dict have the same keys
        for key, value_dict in stats.items():
            stats_dict[key] = _welfords_online(value_dict, stats_dict[key])

    # deviation to std
    for v in stats_dict.values():
        v['std'] = math.sqrt((v['accum_m2'] / (v['total_count']-1)))

    # save results to file and return
    logger.log('INFO', f'Global image stats save to {stats_fpath}')
    utils.write_json(stats_fpath, stats_dict)
    utils.hash_artifacts(stats_fpath)
    return stats_dict

def _validate_image_stats(
    blks_fpaths: list[str],
    logger: utils.Logger,
) -> bool:
    '''doc'''

    jobs = [(_check_block_image_stats, (f, ), {}) for f in blks_fpaths]
    rr: list[dict] = utils.ParallelExecutor().run(jobs)
    to_fix = [r.get('invalid', 0) for r in rr if r.get('invalid', 0)]
    # fix if any blocks have invalida stats
    if not to_fix:
        logger.log('INFO', 'All blocks have complete stats')
        return True
    logger.log('INFO', f'Found {len(to_fix)} blocks with bad stats')
    for fpath in tqdm.tqdm(to_fix):
        rb = blockbuilder.DataBlock().load(fpath)
        rb.recompute_image_stats(fpath) # save to overwrite
    logger.log('INFO', f'Updated stats for {len(to_fix)} blocks')
    return False

def _check_block_image_stats(block_fpath: str) -> dict[str, str]:
    '''doc'''

    meta = blockbuilder.DataBlock().load(block_fpath).meta
    stats = meta['block_image_stats']
    for value_dict in stats.values():
        if any(numpy.isnan(x) for x in value_dict.values()):
            return {'invalid': block_fpath}
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
    mt += delta * nb / (nt + nb)
    dt += m2b + (delta ** 2) * nt * nb / (nt + nb)
    nt += nb
    # assign back and return
    current_results['total_count'] = nt
    current_results['current_mean'] = mt
    current_results['accum_m2'] = dt
    return current_results
