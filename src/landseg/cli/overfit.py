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

    # build a trainer with minimal settings
    config.models['conditioning']['mode'] = 'none'
    config.trainer['optim']['weight_decay'] = 0.0
    config.trainer['runtime']['schedule']['log_every'] = 1
    config.trainer['runtime']['optimization']['grad_clip_norm'] = None
    config.trainer['loss']['types']['focal']['weight'] = 1.0
    config.trainer['loss']['types']['focal']['gamma'] = 0.0
    config.trainer['loss']['types']['dice']['weight'] = 0.0
    trainer = training.build_trainer(dataspecs, config, logger)
    trainer.set_head_state(['layer1'])

    # run trainer
    max_epoch = config['overfit_test_max_epoch']
    logger.log('INFO', f'Starting overfit test for maximum {max_epoch} epochs')
    for ep in range(1, max_epoch + 1):
        los = trainer.train_one_epoch(ep)['Total_Loss']
        iou = trainer.validate()['layer1']['mean']
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

    # build a dataspec from schema dict (skipping dataset module)
    dspecs = dataset.build_dataspec_from_a_block(blk_schema) # with dummy values
    return dspecs
