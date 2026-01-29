'''Pipeline to prepare data from rasters for training.'''

# third-party imports
import omegaconf
# local imports
import dataset
import dataset.summary
import utils

def prepare_data(
        dataset_name: str,
        config: omegaconf.DictConfig,
        logger: utils.Logger,
        *,
        skip: bool=False
    ) -> dataset.summary.DataSummary:
    '''Data preparation pipeline.'''

    # if to run the whole process
    if not skip:

        # get all valid blocks
        dataset.build_data_cache(
            dataset_name=dataset_name,
            input_config=config.input,
            cache_config=config.cache,
            logger=logger
        )

        # parse domain knowledge if provided
        dataset.build_domains(
            dataset_name=dataset_name,
            input_config=config.input,
            cache_config=config.cache,
            logger=logger
        )

        # get split datasets for training
        dataset.split_dataset(
            dataset_name=dataset_name,
            cache_config=config.cache,
            logger=logger
        )

        # use stats from training blocks to normalize all valid blocks
        dataset.normalize_datasets(
            dataset_name=dataset_name,
            cache_config=config.cache,
            logger=logger
        )

    # return blocks metadata
    return dataset.summary.generate(
        dataset_name=dataset_name,
        cache_config=config.cache
    )
