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
    logger = utils.Logger('main', os.path.join(test_dir, 'test.log'))

    # create a single test block and derive dataspec for downstream
    test_dir = os.path.join(config['exp_root'], 'results/overfit_test')
    dataspecs = _mock_dataspecs(config, test_dir, logger)

    # build a trainer with minimal extras
    # overfit test settings
    config.models['conditioning']['mode'] = 'none'
    config.trainer['optim']['weight_decay'] = 0.0
    config.trainer['runtime']['schedule']['log_every'] = 1
    config.trainer['runtime']['optimization']['grad_clip_norm'] = None
    config.trainer['loss']['types']['focal']['weight'] = 1.0
    config.trainer['loss']['types']['focal']['gamma'] = 0.0
    config.trainer['loss']['types']['dice']['weight'] = 0.0
    trainer = training.build_trainer(dataspecs, config, logger)
    # trainer.comps.optimization.optimizer.

    # run trainer
    max_epoch = config['overfit_test_max_epoch']
    trainer.set_head_state(['layer1'])
    logger.log('INFO', f'Starting overfit test for maximum {max_epoch} epochs')
    for epoch in range(1, max_epoch + 1):
        loss = trainer.train_one_epoch(epoch)['Total_Loss']
        iou = trainer.validate()['layer1']['mean']
        logger.log('INFO', f'Epoch: {epoch:04d} | Loss: {loss:4f} | IoU: {iou:4f}')
        if iou >= 0.99:
            logger.log('INFO', 'Overfit reached - test complete')
            break

def _mock_dataspecs(
    config: omegaconf.DictConfig,
    test_dir: str,
    logger: utils.Logger
) -> dataset.DataSpecs:
    '''Manually generate a `DataSpecs` instance from a signle block.'''

    # load world grid
    gid, world_grid = grid.prep_world_grid(config.extent, config.grid, logger)

    # build a single block and write to a temporary location
    blk= dataprep.prepare_data(
        (gid, world_grid),
        config.dataset,
        config.artifacts,
        config.dataprep,
        logger,
        build_a_block=True # ensure function returns a class instance
    )
    assert blk # typing sanity
    blk.save(os.path.join(test_dir, 'overfit_test_block.npz'))

    # build a dataspec from schema dict (skipping dataset module)
    dspecs = dataset.build_empty_dataspec() # with dummy values
    # populated dataspecs with essentials
    # metadata
    dspecs.meta.dataset_name = blk.meta['block_name']
    dspecs.meta.img_ch_num = blk.data.image_normalized.shape[0]
    dspecs.meta.ignore_index = blk.meta['ignore_label']
    # heads - all dummy values
    counts = blk.meta['label_count']
    cc = {k: [1] * len(counts[k]) for k in counts if k != 'original_label'}
    dspecs.heads.class_counts = cc
    dspecs.heads.logits_adjust = {k: [1.0] * len(v) for k, v in cc.items()}
    dspecs.heads.topology = _get_topology(blk.meta['label_count'])
    # splits - same block for both training and validation
    dspecs.splits.train = {blk.meta['block_name']: './overfit_test_block.npz'}
    dspecs.splits.val = {blk.meta['block_name']: './overfit_test_block.npz'}
    # return
    return dspecs

def _get_topology(label_count: dict[str, list[int]]):
    '''From `dataprep.schema.py`.'''

    topology: dict[str, dict[str, str | int | None]] = {}
    # iterate through label counts
    for layer_name in label_count:
        if layer_name == 'original_label': # skip this
            continue
        # emit topology for current convention - from layer naming
        if layer_name == 'layer1':
            topology[layer_name] = {'parent': None, 'parent_cls': None}
        elif layer_name.startswith('layer2_'):
            cls_id = int(layer_name.split('layer2_')[1])
            topology[layer_name] = {'parent': 'layer1', 'parent_cls': cls_id}
        else:
            # if future names appear, one can decide to raise or set None
            topology[layer_name] = {'parent': None, 'parent_cls': None}

    # return the dicts
    return topology
