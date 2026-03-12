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
Overfit test routine that builds a single-block dataspec and trains a \
minimal model until convergence.
'''

# standard imports
import os
# local imports
import landseg.configs as configs
import landseg.data_schema as data_schema
import landseg.models as models
import landseg.factory as factory
import landseg.utils as utils

def overfit_test(config: configs.RootConfig) -> None:
    '''Run an overfit test using a single prepared data block.'''

    # create a logger at dedicated folder
    test_dir = os.path.join(config.exp_root, 'results/overfit_test')
    logger = utils.Logger('test', os.path.join(test_dir, 'test.log'))

    # create a single test block and derive dataspec for downstream
    dataspecs = data_schema.load_data(
        config.inputs,
        config.prep,
        logger,
        single_block_mode=True,
        single_block_dir=test_dir
    )

    # parse from config
    monitor_head = config.trainer.runtime.monitor.track_head_name
    max_epoch = config.trainer.runtime.schedule.max_epoch

    # setup the model
    model = models.build_multihead_unet(dataspecs, config.models)

    # build a trainer with no logging
    trainer = factory.build_trainer(
        dataspecs,
        model,
        config.trainer,
        logger,
        skip_log=True
    )
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
