'''
Test
'''

# local imports
import landseg.configs as configs
import landseg.core as core
import landseg._ingest_dataset.canonical as canonical
import landseg._ingest_dataset.materialized as materialized
import landseg.utils as utils

def test_pipeline(
    world_grid: core.GridLayoutLike,
    config: configs.RootConfig,
    logger: utils.Logger
):
    '''Test..'''

    # temp configs
    canon_output_root = './experiment/artifacts/data_cache/branch_test/canonical'
    dataset_output_root = './experiment/artifacts/data_cache/branch_test/dataset'

    # build/update data catalog
    catalog_inputs = canonical.CatalogueInputs(
        fit_image_fpath=config.inputs.data.filepaths.fit_image,
        fit_label_fpath=config.inputs.data.filepaths.fit_label,
        test_image_fpath=config.inputs.data.filepaths.test_image,
        test_label_fpath=config.inputs.data.filepaths.test_label,
        data_config_fpath=config.inputs.data.filepaths.config
    )
    canonical.build_catalogue(
        world_grid,
        catalog_inputs,
        canon_output_root,
        logger,
        single_block_mode=False
    )

    # load catalog from json artifact
    catalog_fpath = f'{canon_output_root}/fit/catalog.json'
    catalog = canonical.BlocksCatalog.from_json(catalog_fpath)

    # materialized to datasets
    dataset_config = materialized.DatasetBuildConfig(
        val_test_ratios=(0.1, 0.1),
        buffer_step=1,
        reward_ratios={2: 5.0, 4: 5.0},
        scoring_alpha=1.0,
        scoring_beta=0.8,
        max_skew_rate=5.0,
        block_spec=(256, 128)
    )
    materialized.build_dataset(
        catalog,
        dataset_config,
        dataset_output_root,
        logger
    )
