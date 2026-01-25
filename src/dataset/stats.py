'''Data blocks preperation pipeline.'''

# standard imports
import math
import os
import random
# third-party imports
import numpy
import tqdm
# local imports
import dataset.blocks
import utils

# --------------------class distribution of each label layer--------------------
def count_label_classes(
        blkslist_fpath: str,
        count_fpath: str,
        *,
        logger: utils.Logger,
        overwrite: bool
    ) -> dict[str, list[int]]:
    '''Aggregate label class counting.'''

    # get a child logger
    logger=logger.get_child('stats')

    # check if already counted
    if os.path.exists(count_fpath) and not overwrite:
        logger.log('INFO', f'Gathering label counts from: {count_fpath}')
        return utils.load_json(count_fpath)

    # aggregate pixel count in each block
    fpaths: list[str] = utils.load_pickle(blkslist_fpath)
    count_results = {}
    for fpath in tqdm.tqdm(fpaths, desc='Counting label class dist.'):
        bb = dataset.blocks.DataBlock().load_from_npz(fpath)
        for layer, counts in bb.meta['label_count'].items():
            bb_count = numpy.asarray(counts)
            if layer in count_results:
                count_results[layer] += bb_count
            else:
                count_results[layer] = bb_count
            count_results[layer] = [int(x) for x in count_results[layer]]

    # save to file and return
    logger.log('INFO', f'Label class distributions save to {count_fpath}')
    utils.write_json(count_fpath, count_results)
    return count_results

# score all valid blocks based on label class distribution
def score_blocks(
        blkscore_fpath: str,
        blkvalid_fpath: str,
        score_param: dict,
        global_count_fpath: str,
        *,
        logger: utils.Logger,
        overwrite: bool
    ) -> list[dict[str, str | int ]]:
    '''doc'''

    # get a child logger
    logger=logger.get_child('stats')

    # check if already scored
    if os.path.exists(blkscore_fpath) and not overwrite:
        logger.log('INFO', f'Gathering block scoring from: {blkscore_fpath}')
        return utils.load_json(blkscore_fpath)

    # load valid block list
    blks = utils.load_pickle(blkvalid_fpath)
    global_count = utils.load_json(global_count_fpath)

    # get scoring parameters with defaults
    layer = score_param.get('layer', 'layer1')
    a = score_param.get('alpha', 0.6)

    # score blocks from list with parallel processing
    target_p  = _count_to_inv_prob(global_count[layer], alpha=a)
    jobs = [(_score_block, (_, target_p, score_param), {}) for _ in blks]
    scores = utils.ParallelExecutor().run(jobs)
    sorted_scores = sorted(scores, key=lambda _: _['score'])

    # save sorted scores to a file
    utils.write_json(blkscore_fpath, sorted_scores)
    return scores

def _score_block(
        block_fpath: str,
        target_p: numpy.ndarray,
        param: dict
    ) -> dict[str, str | int | float | list[float]]:
    '''Score each block.'''

    # read from block
    col, row = dataset.blocks.parse_block_name(block_fpath).colrow
    bb = dataset.blocks.DataBlock().load_from_npz(block_fpath)

    # parse from parameter dict
    layer = param.get('layer', 'layer1')
    b = param.get('beta', 1.0)
    e = param.get('epsilon', 1e-12)
    reward_cls = param.get('reward', [])

    # block class distributions
    block_p = _count_to_inv_prob(bb.meta['label_count'][layer], alpha=1.0)
    # L1 distance to measure pairwise closeness
    score = _weighted_l1_w_reward(target_p, block_p, reward_cls, b, e)

    # return
    return {
        'file_path': block_fpath,
        'col': col,
        'row': row,
        'score': score,
        'block_p': [round(p, 4) for p in block_p],
        'off_target': [round(p, 4) for p in block_p - target_p]
    }

def _count_to_inv_prob(
        counts: list[int],
        alpha: float,
        epsilon: float=1e-12
    ) -> numpy.ndarray:
    '''Global count to inverse distribution with epsilon smoothing.'''

    # safe count to probability distribution
    arr = numpy.asarray(counts)
    tt = arr.sum()
    assert tt > 0
    p = arr / tt
    p = numpy.clip(p, epsilon, 1.0) # avoid downstream log(0)
    p = p / p.sum() # re-normalize to have probability sum==1

    # inverse and return
    inv = p ** (alpha)
    return inv / inv.sum()

def _weighted_l1_w_reward(
        p: numpy.ndarray,
        q: numpy.ndarray,
        reward_cls: list[int],
        beta: float,
        eps: float
    ) -> float:
    '''Weighted L1 distance between distributions with rewards.'''

    # sanity check
    assert numpy.isclose(p.sum(), 1), p.sum()
    assert numpy.isclose(q.sum(), 1), q.sum()
    # p as target q as test
    # use log weight and bonus for reward classes > target classes
    w_l1 = sum(abs(a - b) * (1 + abs(math.log(a + eps))) for a, b in zip(p, q))
    bonus = sum(max(0, q[i] - p[i]) for i in reward_cls) * beta
    return float(w_l1 - bonus)

# -------------global image stats aggregated from selected blocks-------------
def get_image_stats(
        blkslist_fpath: str,
        stats_fpath: str,
        *,
        logger: utils.Logger,
        overwrite: bool
    ) -> dict[str, dict[str, int | float]]:
    '''Aggregate stats from a given list of blocks'''

    # get a child logger
    logger=logger.get_child('stats')

    # check if stats are complete
    logger.log('INFO', 'Checking block image stats')
    block_list: list[str] = utils.load_pickle(blkslist_fpath)
    stats_complete = _validate_image_stats(block_list, logger)

    # load stats if file already exists
    if os.path.exists(stats_fpath) and stats_complete and not overwrite:
        logger.log('INFO', f'Gathering image stats from: {stats_fpath}')
        return utils.load_json(stats_fpath)

    logger.log('INFO', 'Aggregating global image stats')
    # read a random block to get number of image channels
    temp = dataset.blocks.DataBlock()
    temp.load_from_npz(random.choice(block_list))
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
    for fpath in tqdm.tqdm(block_list):
        # prep
        rb = dataset.blocks.DataBlock().load_from_npz(fpath)
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
        block_list: list[str],
        logger: utils.Logger,
    ) -> bool:
    '''doc'''

    jobs = [(_check_block_image_stats, (f, ), {}) for f in block_list]
    rr: list[dict] = utils.ParallelExecutor().run(jobs)
    work_fpaths = [r.get('restats', 0) for r in rr if r.get('restats', 0)]
    # fix if any blocks have invalida stats
    if not work_fpaths:
        logger.log('INFO', 'All blocks have complete stats')
        return True
    logger.log('INFO', f'Found {len(work_fpaths)} blocks with bad stats')
    for fpath in tqdm.tqdm(work_fpaths):
        rb = dataset.blocks.DataBlock().load_from_npz(fpath)
        rb.recalc_stats(fpath) # save to overwrite
    logger.log('INFO', f'Updated stats for {len(work_fpaths)} blocks')
    return False

def _check_block_image_stats(block_fpath: str) -> dict[str, str]:
    '''doc'''

    meta = dataset.blocks.DataBlock().load_from_npz(block_fpath).meta
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

# -------------normalized selected blocks using global image stats-------------
def normalize_blocks(
        blkslist_fpath: str,
        stats_fpath: str,
        *,
        logger: utils.Logger,
        overwrite: bool
    ) -> None:
    '''Normalize blocks using provided global stats dict.'''

    # get a child logger
    logger=logger.get_child('stats')

    block_fpaths: list[str] = utils.load_pickle(blkslist_fpath)

    # get blocks that needs to be updated for image normalization
    if overwrite:
        work_fpaths = block_fpaths
    else:
        # multiprocessing check blocks on normalized image channels
        logger.log('INFO', 'Checking block image normalization')
        jobs = [(_check_block_normal, (f, ), {}) for f in block_fpaths]
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

    data = dataset.blocks.DataBlock().load_from_npz(block_fpath).data
    if data.image_normalized.size != data.image.size or \
        numpy.isnan(data.image_normalized).any():
        return {'renorm': block_fpath}
    return {'passed': block_fpath}

def _normalize_block(
        fpath: str,
        stats: dict,
    ) -> None:
    '''doc.'''

    rb = dataset.blocks.DataBlock().load_from_npz(fpath)
    mmin, mmax = rb.normalize_image(stats)
    assert mmin > -100 and mmax < 100
    rb.save_npz(fpath)
