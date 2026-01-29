'''Select validation dataset pipeline.'''

# standard imports
import math
import os
import re
# third-party imports
import numpy
# local imports
import _types
import dataset
import utils

# public API
def split_dataset(
        dataset_name: str,
        cache_config: _types.ConfigType,
        logger: utils.Logger
    ):
    '''doc'''

    # training blocks dirpath
    training_dir = f'./data/{dataset_name}/cache/training'

    # config accessors
    cache_cfg = utils.ConfigAccess(cache_config)

    # get artifacts names
    lbl_count = cache_cfg.get_asset('artifacts', 'label', 'count_global')
    valid_blks = cache_cfg.get_asset('artifacts', 'blocks', 'valid')

    # artifact filepaths
    lbl_count_fpath = os.path.join(training_dir, lbl_count)
    valid_blks_fpath = os.path.join(training_dir, valid_blks)

    # count label classes globally (all valid blocks)
    dataset.count_label_cls(
        blks_fpaths=valid_blks_fpath,
        results_fpath=lbl_count_fpath,
        logger=logger,
        overwrite=cache_cfg.get_option('flags', 'overwrite_counts')
    )

    # score valid blocks
    score(training_dir, cache_cfg, logger)

    # split datasets according to scores
    split(training_dir, cache_cfg, logger)

# ----------score all valid blocks based on label class distribution----------
def score(
        training_dir: str,
        cache_cfg: utils.ConfigAccess,
        logger: utils.Logger,
    ) -> None:
    '''doc'''

    # get a child logger
    logger = logger.get_child('score')

    # get artifacts names
    lbl_count = cache_cfg.get_asset('artifacts', 'label', 'count_global')
    valid_blks = cache_cfg.get_asset('artifacts', 'blocks', 'valid')
    blks_scores = cache_cfg.get_asset('artifacts', 'scoring', 'by_block')

    # artifact filepaths
    lbl_count_fpath = os.path.join(training_dir, lbl_count)
    valid_blks_fpath = os.path.join(training_dir, valid_blks)
    blks_scores_fpath = os.path.join(training_dir, blks_scores)

    # get scoring paramters
    score_param = cache_cfg.get_section_as_dict('scoring')

    # get overwrite option
    overwrite = cache_cfg.get_option('flags', 'overwrite_scores')

    # check if already scored
    if os.path.exists(blks_scores_fpath) and not overwrite:
        logger.log('INFO', f'Gathering block scoring from: {blks_scores_fpath}')
        return utils.load_json(blks_scores_fpath)

    # load valid block list
    blks: dict[str, str] = utils.load_json(valid_blks_fpath)
    global_count: dict[str, list[int]] = utils.load_json(lbl_count_fpath)

    # score blocks from list with parallel processing
    target_p  = _count_to_inv_prob(global_count, **score_param)
    jobs = [(_score_block, (b, target_p,), score_param) for b in blks.values()]
    scores = utils.ParallelExecutor().run(jobs)
    sorted_scores = sorted(scores, key=lambda _: _['score'])

    # save sorted scores to a file
    utils.write_json(blks_scores_fpath, sorted_scores)

def _score_block(
        block_fpath: str,
        target_p: numpy.ndarray,
        **kwargs
    ) -> dict[str, str | int | float | list[float]]:
    '''Score each block.'''

    # read from block
    meta = dataset.DataBlock().load(block_fpath).meta
    name = meta['block_name']

    # find pattern from string
    pattern = r'col_(\d+)_row_(\d+)'
    matched = re.search(pattern, name)
    # there should be just one match
    if not matched:
        raise ValueError(f'Block naming pattern {pattern} not found')
    # get col and row
    col = int(matched.group(1))
    row = int(matched.group(2))

    # parse from parameter dict
    b = kwargs.get('beta', 1.0)
    e = kwargs.get('epsilon', 1e-12)
    reward_cls = kwargs.get('reward', [])

    # block class distributions
    block_p = _count_to_inv_prob(meta['label_count'], alpha=1.0)
    # L1 distance to measure pairwise closeness
    _score = _weighted_l1_w_reward(target_p, block_p, reward_cls, b, e)

    # return
    return {
        'block_name': name,
        'file_path': block_fpath,
        'col': col,
        'row': row,
        'score': _score,
        'block_p': [round(p, 4) for p in block_p],
        'off_target': [round(p, 4) for p in block_p - target_p]
    }

def _count_to_inv_prob(
        counts: dict[str, list[int]],
        **kwargs
    ) -> numpy.ndarray:
    '''Global count to inverse distribution with epsilon smoothing.'''

    # get parameters from kwargs
    layer = kwargs.get('layer', 'layer1')
    alpha = kwargs.get('alpha', 0.6)
    epsilon = kwargs.get('epsilon', 1e-12)

    # safe count to probability distribution
    arr = numpy.asarray(counts[layer])
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

def split(
        training_dir: str,
        cache_cfg: utils.ConfigAccess,
        logger: utils.Logger,
    ) -> None:
    '''Split validation/training data.'''

    # get a child logger
    logger = logger.get_child('split')

    # get artifacts names
    training_blks = cache_cfg.get_asset('artifacts', 'split', 'training')
    validation_blks = cache_cfg.get_asset('artifacts', 'split', 'validation')
    blks_scores = cache_cfg.get_asset('artifacts', 'scoring', 'by_block')

    # artifact filepaths
    t_fpath = os.path.join(training_dir, training_blks)
    v_fpath = os.path.join(training_dir, validation_blks)
    s_fpath = os.path.join(training_dir, blks_scores)

    # get overwrite option
    overwrite = cache_cfg.get_option('flags', 'overwrite_split')

    # skip if not to overwrite and all files are present
    if os.path.exists(v_fpath) and os.path.exists(t_fpath) and not overwrite:
        logger.log('INFO', 'Keeping existing validation/training dataset split')
        return

    # get validation blocks
    scores: list[dict[str, str | int ]] = utils.funcs.load_json(s_fpath)
    v_blks = _select_validation_blocks(scores, logger)
    t_blks = {
        b['block_name']: b['file_path']
        for b in scores if b['block_name'] not in v_blks
    }

    # pickle validation blocks to a file
    utils.write_json(v_fpath, v_blks)
    # pickle training blocks to a file
    utils.write_json(t_fpath, t_blks)

def _select_validation_blocks(
        scores: list[dict[str, str | int ]] ,
        logger: utils.Logger,
        **kwargs
    ) -> dict[str, str]:
    '''Select a set of validation blocks.'''

    valblk_per = kwargs.get('toprange', 0.1)
    min_dist = kwargs.get('mindist', 400)

    # get an expanded top ranking blocks
    num_blk = round(len(scores) * valblk_per)
    existing_locs = [(0, 0)] # init the list
    return_dict = {}

    # iterat through blocks
    for i, blk in enumerate(scores):
        x = blk['col']
        y = blk['row']
        assert isinstance(x, int) and isinstance(y, int) # typing sanity
        dd = _min_distance_from_locs(existing_locs, (x, y))
        if dd >= min_dist:
            existing_locs.append((x, y))
            return_dict[blk['block_name']] = blk['file_path']
        if len(return_dict) == num_blk:
            logger.log('INFO', f'Gathered enough blocks at block {i + 1}')
            break
    logger.log('INFO', f'Gathered {len(return_dict)} blocks from all')

    # return
    return return_dict

def _min_distance_from_locs(
        existing_locs: list[tuple[int, int]],
        input_locs: tuple[int, int]
    ) -> float:
    '''Min Euclidean distance between input and existing locations.'''

    distances = []
    xx, yy = input_locs
    for x, y in existing_locs:
        distances.append(math.sqrt((xx - x) ** 2 + (yy - y) ** 2))
    if 0 in distances:
        distances.remove(0)
    return min(distances)
