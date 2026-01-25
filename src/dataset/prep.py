'''Pipeline to prepare data from rasters for training.'''

# standard imports
import os
# third-party imports
import omegaconf
# local imports
import dataset.blocks
import dataset.domain
import dataset.split
import dataset.stats
import dataset.summary
import utils

def build_data_cache(
        config: omegaconf.DictConfig,
        logger: utils.Logger
    ) -> None:
    '''Create cache blocks from scratch'''

    # create cache dirs
    os.makedirs(config.paths.cache, exist_ok=True)
    os.makedirs(config.paths.blksdpath, exist_ok=True)

    # collect paths for tiling
    paths = dataset.blocks.CachePaths(
        image_fpath=config.inputs.image,
        label_fpath=config.inputs.label,
        meta_fpath=config.inputs.meta,
        blks_dpath=config.paths.blksdpath,
        blk_scheme=config.paths.blkscheme,
        valid_blks=config.paths.blkvalid
    )

    # gather config
    blk_config = dataset.blocks.CacheConfig(
        blk_size=config.blocks.size,
        overlap=config.blocks.overlap,
        valid_px_threshold=config.filters.pxthres,
        water_px_threshold=config.filters.watthres
    )

    # gather blocks from raw inputs
    dataset.blocks.tile_rasters(
        paths=paths,
        config=blk_config,
        logger=logger,
        overwrite=config.overwrite.scheme
    )

    # create block caches - first step, no image normalization
    dataset.blocks.create_block_cache(
        paths=paths,
        logger=logger,
        run_cleanup=config.cleanup.clean_npz,
        overwrite=config.overwrite.cache
    )

    # filter valid blocks and create a list of npz files
    dataset.blocks.validate_blocks_cache(
        paths=paths,
        config=blk_config,
        logger=logger,
        overwrite=config.overwrite.valid
    )

def parse_domain(
        config: omegaconf.DictConfig,
        domain_config: list[dict] | None,
        logger: utils.Logger
    ) -> None:
    '''Parse domain knowledge if provided.'''

    dataset.domain.parse(
        scheme_fpath=config.paths.blkscheme,
        valid_fpath=config.paths.blkvalid,
        domain_fpath=config.paths.domain,
        domain_config=domain_config,
        logger=logger,
        overwrite=config.overwrite.domain
    )

def split_datasets(
        config: omegaconf.DictConfig,
        score_params: dict,
        valselect_param: dict,
        logger: utils.Logger
    ) -> None:
    '''Split blocks into train/validation/test sets.'''

    # count classes from all blocks
    dataset.stats.count_label_classes(
        blkslist_fpath=config.paths.blkvalid,
        count_fpath=config.paths.lblcountg, # ..g for gloabl,
        logger=logger,
        overwrite=config.overwrite.count
    )

    # score all blocks by class distribution from selected layer
    dataset.stats.score_blocks(
        blkscore_fpath=config.paths.blkscore,
        blkvalid_fpath=config.paths.blkvalid,
        score_param=score_params,
        global_count_fpath=config.paths.lblcountg,
        logger=logger,
        overwrite=config.overwrite.score
    )

    dataset.split.run(
        scores_fpath=config.paths.blkscore,
        v_fpath=config.paths.dataval,
        t_fpath=config.paths.datatrain,
        valselect_param=valselect_param,
        logger=logger,
        overwrite=config.overwrite.split
    )

def normalize_datasets(
        config: omegaconf.DictConfig,
        logger: utils.Logger
    ) -> None:
    '''Aggragate block stats'''

    # count classes from training blocks
    dataset.stats.count_label_classes(
        blkslist_fpath=config.paths.datatrain,
        count_fpath=config.paths.lblcountt, # ..t for training
        logger=logger,
        overwrite=config.overwrite.count
    )

    # aggregate stats from training blocks
    dataset.stats.get_image_stats(
        blkslist_fpath=config.paths.datatrain,
        stats_fpath=config.paths.imgstats,
        logger=logger,
        overwrite=config.overwrite.stats
    )

    # normalize all valid blocks using the aggregated stats from training data
    dataset.stats.normalize_blocks(
        blkslist_fpath=config.paths.blkvalid,
        stats_fpath=config.paths.imgstats,
        logger=logger,
        overwrite=config.overwrite.norm
    )

def validate_config(config: omegaconf.DictConfig):
    '''doc'''

    dom = omegaconf.OmegaConf.to_container(config.inputs.domain, resolve=True)
    if dom is not None:
        assert isinstance(dom, list)
        assert all(isinstance(x, dict) for x in dom)

    score = omegaconf.OmegaConf.to_container(config.scoring, resolve=True)
    assert isinstance(score, dict)

    val = omegaconf.OmegaConf.to_container(config.valselect, resolve=True)
    assert isinstance(val, dict)

    return dom, score, val

def run(
        config: omegaconf.DictConfig,
        logger: utils.Logger
    ) -> dataset.summary.DataSummary:
    '''Data preparation pipeline.'''

    dom_cfg, score_params, valselect = validate_config(config)

    # if to run the whole process
    if not config.skip_dataprep:

        # get all valid blocks
        build_data_cache(config, logger)

        # parse domain knowledge if provided
        parse_domain(config, dom_cfg, logger)

        # get training blocks
        split_datasets(config, score_params, valselect, logger)

        # use stats from training blocks to normalize all valid blocks
        normalize_datasets(config, logger)

    # return blocks metadata
    return dataset.summary.generate(
        validblks_fpath=config.paths.blkvalid,
        train_lblstats_fpath=config.paths.lblcountt,
        train_datablks_fpaths=config.paths.datatrain,
        val_datablks_fpaths=config.paths.dataval,
        domain_fpath=config.paths.domain
    )
