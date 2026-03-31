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

# local imports
import landseg.configs as configs
import landseg.core as core
import landseg.geopipe.core as geocore
import landseg.geopipe.foundation as foundation
import landseg.models as models
import landseg.trainer as trainer
import landseg.utils as utils

def overfit(config: configs.RootConfig):
    '''
    Run an overfit test on a single block.

    Creates or loads a block, builds a small `DataSpecs`, instantiates
    the model and a trainer with minimal logging, and trains until an IoU
    threshold or the epoch limit is reached.

    Args:
        config: RootConfig with model/trainer settings.
    '''

    # root dpath
    root = f'{config.exp_root}/results/overfit_test'

    # init a logger
    logger = utils.Logger('overfit', f'{root}/log')

    # get a single block
    block_fp = _build_a_block(config, root, logger)

    # build the dataspecs dataclass
    dataspecs = _build_dataspec_a_block(block_fp)

    # setup the model
    model = models.build_multihead_unet(dataspecs, config.models)

    # build a trainer with no logging
    monitor_head = config.trainer.runtime.monitor.track_head_name
    _trainer = trainer.build_trainer(
        dataspecs,
        model,
        config.trainer,
        logger,
        skip_log=True
    )
    _trainer.set_head_state([monitor_head])

    # run trainer
    max_epoch = config.trainer.runtime.schedule.max_epoch
    logger.log('INFO', f'Starting overfit test for maximum {max_epoch} epochs')
    for ep in range(1, max_epoch + 1):
        los = _trainer.train_one_epoch(ep)['Total_Loss']
        iou = _trainer.validate()[monitor_head]['mean']
        logger.log('INFO', f'Epoch: {ep:04d} | Loss: {los:4f} | IoU: {iou:4f}')
        if iou >= 0.99:
            logger.log('INFO', 'Overfit reached - test complete')
            break

def _build_a_block(
    config: configs.RootConfig,
    save_dpath: str,
    logger: utils.Logger
) -> str:
    '''Build or select one valid block for the overfit test.'''

    # config aliases
    grid_cfg = config.foundation.grid
    datablocks_cfg = config.foundation.datablocks
    out_root = config.foundation.output_dpath

    # prep world grid
    _config = foundation.GridExtentConfig(
        mode=grid_cfg.mode, # type: ignore
        crs=grid_cfg.crs,
        ref_fpath=grid_cfg.extent.filepath,
        origin=grid_cfg.extent.origin,
        pixel_size=grid_cfg.extent.pixel_size,
        grid_extent=grid_cfg.extent.grid_extent,
        grid_shape=grid_cfg.extent.grid_shape
    )
    grid_gen_config = foundation.GridGenerationConfig(
        output_dir=f'{out_root}/world_grids',
        tile_size=(grid_cfg.tile_size.row, grid_cfg.tile_size.col),
        tile_overlap=(grid_cfg.tile_overlap.row, grid_cfg.tile_overlap.col)
    )
    grid = foundation.build_world_grid(_config, grid_gen_config, logger)

    # build a single block
    _config = foundation.BlockBuildingParameters(
        dev_image_fpath=datablocks_cfg.filepaths.dev_image,
        dev_label_fpath=datablocks_cfg.filepaths.dev_label,
        test_image_fpath=datablocks_cfg.filepaths.test_image,
        test_label_fpath=datablocks_cfg.filepaths.test_label,
        data_config_fpath=datablocks_cfg.filepaths.config,
        dem_pad=datablocks_cfg.general.image_dem_pad,
        ignore_index=datablocks_cfg.general.ignore_index,
    )
    blocks_dir = f'{out_root}/data_blocks'
    block_fpath = foundation.run_blocks_building(
        grid,
        _config,
        blocks_dir,
        logger,
        single_block_mode=True,
        save_dpath=save_dpath,
        valid_px_per=0.8,
        monitor_head='base',
        need_all_classes=True
    )
    if not block_fpath:
        raise ValueError('No valid block for testing is found')
    return block_fpath

def _build_dataspec_a_block(block_fpath: str) -> core.DataSpecs:
    '''Build a minimal `DataSpecs` from a single saved block.'''

        # read bgeocore
    block = geocore.DataBlock.load(block_fpath)
    counts = block.meta['label_count']
    cc = {k: [1] * len(counts[k]) for k in counts if k != 'original'}

    # returgeocorerectly from schema dict
    return core.DataSpecs(
        name='',
        mode='single',
        meta =core.Meta(
            img_ch=block.data.image.shape[0],
            img_h_w=block.data.label.shape[1], # here assume H==W
            ignore_index=block.meta['ignore_index'],
            img_arr_key='image', # as per convention (already normalized)
            lbl_arr_key='label_stack', # as per convention (unchanged)
            blk_bytes=0,
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
