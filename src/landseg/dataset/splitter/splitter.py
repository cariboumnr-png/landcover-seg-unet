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
Dataset splitting utilities for producing train/val block lists.
Orchestrates global label counting, block scoring, validation selection,
and artifact persistence.

Public APIs:
    - split_blocks: Create train/val splits from scored blocks and write
      results to artifacts.
'''

# standard imports
import os
# local imports
import landseg.dataset as dataset
import landseg.dataset.splitter as splitter
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def split_blocks(
    process_config: dataset.ProcessConfig,
    logger: utils.Logger,
    *,
    rebuild: bool = False
) -> None:
    '''
    'Create train/val splits from scored blocks and persist artifacts.

    Args:
        process_config: Process configuration containing input artifact
            paths (e.g., fit_valid_blks) and output targets (e.g.,
            train_blks, val_blks, lbl_count_*), plus scoring parameters.
        logger: Logger used for progress and status reporting.
        rebuild: If True, overwrite existing split artifacts; if False,
            keep existing train/val files when both are present.
    '''

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
    # WIP here simply use the rest without spatial buffering and exclusion
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
