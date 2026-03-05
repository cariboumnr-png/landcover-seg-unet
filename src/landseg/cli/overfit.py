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

'''Overfit test on a single data block.'''

# standard imports
import os
# third-party imports
import omegaconf
# local imports
import landseg.dataprep as dataprep
import landseg.dataset as dataset
import landseg.grid as grid
import landseg.training as training
import landseg.utils as utils

def overfit_test(config: omegaconf.DictConfig) -> None:
    '''Overfit test on a single data block.'''

    # create a logger at dedicated folder
    test_dir = os.path.join(config['exp_root'], 'results/overfit_test')
    logger = utils.Logger('test', os.path.join(test_dir, 'test.log'))

    # create a single test block and derive dataspec for downstream
    dataspecs = _single_block_dataspecs(config, test_dir, logger)

    # parse from config
    monitor_head = config['trainer']['runtime']['monitor']['track_head_name']
    max_epoch = config['trainer']['runtime']['schedule']['max_epoch']

    # build a trainer with minimal settings
    trainer = training.build_trainer(dataspecs, config, logger)
    trainer.set_head_state([monitor_head])

    # run trainer
    logger.log('INFO', f'Starting overfit test for maximum {max_epoch} epochs')
    for ep in range(1, max_epoch + 1):
        los = trainer.train_one_epoch(ep)['Total_Loss']
        iou = trainer.validate()[monitor_head]['mean']
        logger.log('INFO', f'Epoch: {ep:04d} | Loss: {los:4f} | IoU: {iou:4f}')
        if iou >= 0.99:
            logger.log('INFO', 'Overfit reached - test complete')
            break

def _single_block_dataspecs(
    config: omegaconf.DictConfig,
    test_dir: str,
    logger: utils.Logger
) -> dataset.DataSpecs:
    '''Manually generate a `DataSpecs` instance from a signle block.'''

    # load world grid
    world_grid = grid.prep_world_grid(config.extent, config.grid, logger)

    # build a minimul schema dict from a single block
    blk_path = os.path.join(test_dir, 'overfit_test_block.npz')
    blk_schema = dataprep.prepare_data(
        world_grid,
        config.dataset,
        config.artifacts,
        config.dataprep,
        logger,
        build_a_block=True,
        block_fpath=blk_path
    )
    assert blk_schema # sanity

    # build a dataspec from the schema with essential values
    dspecs = dataset.build_dataspec_from_a_block(blk_schema)
    return dspecs
