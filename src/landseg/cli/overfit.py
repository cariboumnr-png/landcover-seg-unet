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

    # create a centralized main logger
    log_dir = os.path.join(config['exp_root'], 'results', 'overfit_test', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    t_stamp = utils.get_timestamp()
    logger = utils.Logger('main', os.path.join(log_dir, f'main_{t_stamp}.log'))

    # create a single test block and derive dataspec for downstream
    dataspecs = _mock_dataspecs(config, logger)

    # build a trainer with minimal extras
    # overfit test settings
    config.models['conditioning']['mode'] = 'none'
    config.trainer['optim']['weight_decay'] = 0.0
    config.trainer['runtime']['schedule']['val_every'] = None
    config.trainer['runtime']['optimization']['grad_clip_norm'] = None
    config.trainer['loss']['types']['focal']['weight'] = 1.0
    config.trainer['loss']['types']['focal']['gamma'] = 0.0
    config.trainer['loss']['types']['dice']['weight'] = 0.0
    trainer = training.build_trainer(dataspecs, config, logger)
    # trainer.comps.optimization.optimizer.
    trainer.flags = {
        'enable_train_la': False,
        'enable_val_la': False,
        'enable_test_la': False
    }
    trainer.set_head_state(active_heads=['layer1'], excluded_cls={'layer1': (5, 6)})

    # use trainer to run
    for epoch in range(1, 3000):
        trainer.train_one_epoch(epoch)

    # remove test block
    os.remove('./overfit_test_block.npz')

def _mock_dataspecs(
    config: omegaconf.DictConfig,
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
    blk.save('./overfit_test_block.npz')

    # build a dataspec from schema dict (skipping dataset module)
    dspecs = dataset.build_empty_dataspec() # with dummy values
    # populated dataspecs with essentials
    # metadata
    dspecs.meta.dataset_name = blk.meta['block_name']
    dspecs.meta.img_ch_num = blk.data.image_normalized.shape[0]
    dspecs.meta.ignore_index = blk.meta['ignore_label']
    # heads
    counts = blk.meta['label_count']
    cc = {k: [1] * len(counts[k]) for k in counts if k != 'original_label'}
    dspecs.heads.class_counts = cc
    dspecs.heads.logits_adjust = {k: [1.0] * len(v) for k, v in cc.items()}
    dspecs.heads.topology = _get_topology(blk.meta['label_count'])
    # splits
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
