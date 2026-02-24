'''doc'''

# standard imports
import os
# local imports
import landseg.dataprep as dataprep
import landseg.dataprep.splitter as splitter
import landseg.utils as utils

def split_blocks(
    process_config: dataprep.ProcessConfig,
    logger: utils.Logger,
    *,
    rebuild: bool = False
) -> None:
    '''doc'''

    # get a child logger
    logger = logger.get_child('split')

    # skip if not to overwrite and all files are present
    if os.path.exists(process_config['train_blks']) and \
        os.path.exists(process_config['val_blks']) and not rebuild:
        logger.log('INFO', 'Keeping existing train/val dataset split')
        return

    # load/create global label class count
    global_cls_count = splitter.count_label_class(
        input_blocks=process_config['fit_valid_blks'],
        output_fpath=process_config['lbl_count_global'],
        logger=logger
    )

    # score the blocks
    score_params = splitter.ScoreParams(
        head=process_config['score_head'],
        alpha=process_config['score_alpha'],
        beta=process_config['score_beta'],
        eps=process_config['score_epsilon'],
        reward_cls=process_config['score_reward']
    )
    scores = splitter.score_blocks(
        global_cls_count=global_cls_count,
        input_blocks=utils.load_json(process_config['fit_valid_blks']),
        params=score_params,
        scores_path=process_config['blk_scores']
    )

    # split blocks by scores
    # get validation blocks
    vals = splitter.select_val_blocks(scores, logger)
    # WIP here simple use the rest without spatial buffering and exclusion
    trains = {b.name: b.path for b in scores if b.name not in vals}

    # write train and val blocks to json artifacts
    utils.write_json(process_config['train_blks'], trains)
    utils.hash_artifacts(process_config['train_blks'])
    utils.write_json(process_config['val_blks'], vals)
    utils.hash_artifacts(process_config['val_blks'])

    # count label classes from training blocks
    splitter.count_label_class(
        input_blocks=process_config['train_blks'],
        output_fpath=process_config['lbl_count_train'],
        logger=logger
    )
