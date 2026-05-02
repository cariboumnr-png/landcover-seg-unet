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
Overfit test pipeline.

Constructs a single valid block, builds minimal data specifications,
and trains until near-perfect IoU to validate the end-to-end stack.
'''

# standard imports
import os
# local imports
import landseg._constants as c
import landseg.configs as configs
import landseg.core as core
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.world_grids as world_grids
import landseg.geopipe.foundation.data_blocks as data_blocks
import landseg.geopipe.foundation.data_blocks.mapper as mapper
import landseg.models as models
import landseg.session as session
import landseg.utils as utils

def overfit(config: configs.RootConfig) -> None:
    '''
    Run an overfit test on a single block.

    Creates or loads a block, builds a small `DataSpecs`, instantiates
    the model and a trainer with minimal logging, and trains until an IoU
    threshold or the epoch limit is reached.

    Args:
        config: RootConfig with model/trainer settings.
    '''

    # root dpath
    root = f'{config.execution.exp_root}/results/overfit_test'

    # init a logger
    logger = utils.Logger('overfit', f'{root}/log')

    # get a single block
    block_fp = _build_a_block(root, config, logger=logger)

    # build the dataspecs dataclass
    dataspecs = _build_dataspec_a_block(block_fp)

    # setup the model
    model = models.build_multihead_unet(
        dataspecs=dataspecs,
        backbone_config=config.models.body_registry[config.models.use_body],
        conditioning=config.models.conditioning,
        enable_logit_adjust=config.models.flags.enable_logit_adjust,
        enable_clamp=config.models.flags.enable_clamp,
        clamp_range=config.models.clamp_range
    )

    # build the session
    session_context=session.SessionBuildContext(device=c.DEVICE)
    runner = session.factory.build_overfit_session(
        dataspecs=dataspecs,
        model=model,
        config=config.session,
        context=session_context,
        logger=logger
    )

    # set monitor head
    monitor_head = config.session.runtime.monitor.track_heads
    runner.set_head_state(monitor_head)

    # run train-evaluate
    max_epoch = config.session.runtime.schedule.max_epoch
    lr = config.session.components.optimization.lr
    logger.log('INFO', 'Starting overfit test')
    logger.log('INFO', f'Maximum epoch: {max_epoch}')
    logger.log('INFO', f'Learning rate: {lr}')
    for ep in range(1, max_epoch + 1):
        results = runner.run_epoch(ep)
        assert results.training and results.validation # typing
        los = results.training.total_loss
        iou = results.validation.target_metrics
        logger.log('INFO', f'Epoch: {ep:04d} | Loss: {los:4f} | IoU: {iou:4f}')
        if iou >= 0.99:
            logger.log('INFO', 'Overfit reached - test complete')
            return

    # if overfit not reached
    hint = 'pipeline=diagnose-overfit session.runtime.schedule.max_epoch=<value>'
    logger.log(
        'WARNING',
        f'IoU did not reach 99% after {max_epoch} epochs. '
        f'Increase the limit via: {hint}'
    )

    # close logger
    logger.close()

def _build_a_block(
    save_dpath: str,
    config: configs.RootConfig,
    *,
    logger: utils.Logger,
    **kwargs
) -> str:
    '''Build or select one valid block for the overfit test.'''

    # early return if there is already a block, e.g., an .npz file
    for f in os.listdir(save_dpath):
        if f.endswith('.npz'):
            block_fpath = os.path.join(save_dpath, f)
            logger.log('INFO', f'Using existing block" {block_fpath}')
            return block_fpath

    logger.log('INFO', 'Preparing world grid')
    # world grid
    grid_cfg = config.foundation.grid
    grid_config = world_grids.GridParameters(
        mode=grid_cfg.mode, # type: ignore
        crs=grid_cfg.crs,
        ref_fpath=grid_cfg.extent.filepath,
        origin=grid_cfg.extent.origin,
        pixel_size=grid_cfg.extent.pixel_size,
        grid_extent=grid_cfg.extent.grid_extent,
        grid_shape=grid_cfg.extent.grid_shape,
        tile_specs=grid_cfg.tile_specs_tuple,
    )
    world_grid = world_grids.build_grid(grid_config)

    # map image unto world grid
    logger.log('INFO', 'Mapping image unto the world grid')
    datablocks_cfg = config.foundation.datablocks
    mapped = mapper.map_rasters(
        world_grid,
        datablocks_cfg.filepaths.dev_image,
        datablocks_cfg.filepaths.dev_label,
        logger=logger
    )

    logger.log('INFO', 'Building a single data block')
    # search windows and build a single block
    builder_config = data_blocks.BlockBuilderConfig(
        image_fpath=datablocks_cfg.filepaths.dev_image,
        label_fpath=datablocks_cfg.filepaths.dev_label,
        config_fpath=datablocks_cfg.filepaths.config,
        output_root=save_dpath,
        dem_pad_px=datablocks_cfg.general.image_dem_pad,
        ignore_index=datablocks_cfg.general.ignore_index,
        block_size=mapped.tile_shape
    )
    block_builder = data_blocks.BlockBuilder(
        mapped.image,
        mapped.label,
        builder_config,
        logger=logger,
    )
    block_fpath = block_builder.build_single_block(
        save_dpath,
        valid_px_per=kwargs.get('valid_px_per', 0.8),
        monitor_head=kwargs.get('monitor_head', 'base'),
        need_all_classes=kwargs.get('need_all_classes', True)
    )
    if not block_fpath:
        raise ValueError('No valid block for testing is found')

    logger.log('INFO', f'Single block successfully created: {block_fpath}')
    return block_fpath

def _build_dataspec_a_block(block_fpath: str) -> core.DataSpecs:
    '''Build a minimal `DataSpecs` from a single saved block.'''

    # read the block
    block = geo_core.DataBlock.load(block_fpath)
    counts = block.meta['label_count']
    cc = {k: [1] * len(counts[k]) for k in counts if k != 'original'}

    # returgeocorerectly from schema dict
    specs = core.DataSpecs(
        name='',
        mode='single',
        meta =core.Meta(
            blk_bytes=0,
            test_blks_grid=(0, 0),
            image_specs=core.Meta.Image(
                num_channels=block.data.image.shape[0],
                height_width=block.data.label.shape[1], # here assume H==W
                array_key='image',
                band_map=block.meta['image_band_map'],
            ),
            label_specs=core.Meta.Label(
                array_key='label_stack',
                ignore_index=block.meta['ignore_index']
            )
        ),
        heads=core.Heads(
            class_counts=cc, # neutral
            logits_adjust={k: [1.0] * len(v) for k, v in cc.items()}, # neutral
            head_parent=block.meta['label_ch_parent'],
            head_parent_cls=block.meta['label_ch_parent_cls'],
        ),
        splits=core.Splits(
            train={block.meta['block_name']: block_fpath},
            val={block.meta['block_name']: block_fpath},
            test={}
        ),
        domains=core.Domains(
            train={'ids_domain': None, 'vec_domain': None},
            val={'ids_domain': None, 'vec_domain': None},
            test={'ids_domain': None, 'vec_domain': None},
            ids_max=-1,
            vec_dim=0
        )
    )
    return specs
