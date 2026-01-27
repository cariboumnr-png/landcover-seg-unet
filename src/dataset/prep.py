'''Pipeline to prepare data from rasters for training.'''

# third-party imports
import omegaconf
# local imports
import dataset.blocks
import dataset.domain
import dataset.split
import dataset.stats
import dataset.summary
import utils

def split_datasets(
        config: omegaconf.DictConfig,
        score_params: dict,
        valselect_param: dict,
        logger: utils.Logger
    ) -> None:
    '''Split blocks into train/validation/test sets.'''

    # count classes from all blocks
    dataset.stats.count_label_classes(
        validblk_json=config.paths.blkvalid,
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
        validblk_json=config.paths.datatrain,
        count_fpath=config.paths.lblcountt, # ..t for training
        logger=logger,
        overwrite=config.overwrite.count
    )

    # aggregate stats from training blocks
    dataset.stats.get_image_stats(
        valid_blks_json=config.paths.datatrain,
        stats_fpath=config.paths.imgstats,
        logger=logger,
        overwrite=config.overwrite.stats
    )

    # normalize all valid blocks using the aggregated stats from training data
    dataset.stats.normalize_blocks(
        blks_fpaths_json=config.paths.blkvalid,
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

    domain_config, score_params, valselect = validate_config(config)

    # if to run the whole process
    if not config.skip_dataprep:

        # get all valid blocks
        dataset.blocks.build_data_cache(config, logger)

        # parse domain knowledge if provided
        dataset.domain.parse(config, domain_config, logger)

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
