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
import landseg.artifacts as artifacts
import landseg.geopipe.foundation.data_blocks.assembler as assembler
import landseg.geopipe.foundation.data_blocks.mapper as mapper
import landseg.geopipe.utils as geo_utils
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
        patch_size=config.session.data_loader.patch_size,
        dataspecs=dataspecs,
        unet_backbone_config=config.models.unet_backbone_config,
        conditioning_config=config.models.conditioning_config,
        enable_clamp=config.models.numeric_safety.enable_clamp,
        clamp_range=config.models.numeric_safety.clamp_range
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
    monitor_head = config.session.orchestration.monitor.track_heads
    assert monitor_head
    runner.set_head_state(list(monitor_head.keys()))

    # run train-evaluate
    max_epoch = c.OVERFIT_MAX_EPOCH
    lr = config.session.engine_optim.lr
    logger.log('INFO', 'Starting overfit test')
    logger.log('INFO', f'Maximum epoch: {max_epoch}')
    logger.log('INFO', f'Learning rate: {lr}')
    for ep in range(1, max_epoch + 1):
        results = runner.run_epoch(ep)
        assert results.training and results.validation # typing
        los = results.training.total_objective
        iou = results.target_metrics
        logger.log('INFO', f'Epoch: {ep:04d} | Loss: {los:4f} | IoU: {iou:4f}')
        if iou >= 0.99:
            logger.log('INFO', 'Overfit reached - test complete')
            logger.close()
            return

    # if overfit not reached
    logger.log('WARNING',f'IoU did not reach 99% after {max_epoch} epochs. ')
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

    # get world grid
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
    )

    logger.log('INFO', 'Building a single data block')
    # search windows and build a single block
    # load dataset config JSON
    ctrl = artifacts.Controller[dict].load_json_or_fail(datablocks_cfg.filepaths.config)
    ctrl.hash(overwrite=False)
    dataset_config = ctrl.fetch()
    assert dataset_config

    # map coordinate names to RasterReadInput objects
    inputs_map = {}
    for coord in mapped.image:
        name = geo_utils.xy_name(coord)
        inputs_map[name] = assembler.RasterReadInput(
            image_fpath=datablocks_cfg.filepaths.dev_image,
            image_window=mapped.image[coord],
            image_band_map=dataset_config['image_band_map'],
            image_dem_pad_px=datablocks_cfg.general.image_dem_pad,
            label_fpath=datablocks_cfg.filepaths.dev_label,
            label_window=mapped.label[coord] if mapped.label else None,
            label_specs=dataset_config.get('label_specs'),
        )

    # build a single block matching criteria for testing
    block_fpath = assembler.build_test_block(
        save_dpath=save_dpath,
        inputs=inputs_map,
        target_head=kwargs.get('monitor_head', 'base'),
        valid_px_per=kwargs.get('valid_px_per', 0.8),
        need_all_classes=kwargs.get('need_all_classes', True),
    )
    if not block_fpath:
        raise ValueError('No valid block for testing is found')

    logger.log('INFO', f'Single block successfully created: {block_fpath}')
    return block_fpath

def _build_dataspec_a_block(block_fpath: str) -> core.DataSpecs:
    '''Build a minimal `DataSpecs` from a single saved block.'''

    # read the block
    block = geo_core.DataBlock.load(block_fpath)
    counts = block.manifest['label_count']
    cc = {k: [1] * len(counts[k]) for k in counts if k != 'original'}

    # returgeocorerectly from schema dict
    specs = core.DataSpecs(
        name='',
        mode='single',
        meta =core.Meta(
            blk_bytes=0,
            test_blks_grid=(0, 0),
            label_color_map=None,
            image_specs=core.Meta.Image(
                num_channels=block.data.image.shape[0],
                height_width=block.data.label.shape[1], # here assume H==W
                array_key='image',
                band_map=block.manifest['image_band_map'],
            ),
            label_specs=core.Meta.Label(
                array_key='label_stack',
                ignore_index=block.manifest['ignore_index']
            )
        ),
        heads=core.Heads(
            class_counts=cc, # neutral
            logits_adjust={k: [1.0] * len(v) for k, v in cc.items()}, # neutral
            head_parent=block.manifest['label_parent'],
            head_parent_cls=block.manifest['label_parent_cls'],
        ),
        splits=core.Splits(
            train={block.manifest['block_name']: block_fpath},
            val={block.manifest['block_name']: block_fpath},
            test={}
        ),
        domains=core.Domains(
            train={'ids_domain': None, 'vec_domain': None},
            val={'ids_domain': None, 'vec_domain': None},
            test={'ids_domain': None, 'vec_domain': None},
            ids_num=-1,
            vec_dim=0
        )
    )
    return specs
