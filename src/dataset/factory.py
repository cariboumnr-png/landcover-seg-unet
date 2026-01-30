'''Pipeline to prepare data from rasters for training.'''

# standard imports
import functools
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
        mode: str
    ) -> dataset.summary.DataSummary:
    '''Data preparation pipeline.'''

    # if skip data preperation, e.g., continue training on existing data
    if mode == 'skip':
        # run data summary and return
        summary = dataset.summary.generate(dataset_name, config.cache)
        return summary

    # otherwise prepare partially filed building functions
    build_data_cache = functools.partial(
        dataset.build_data_cache,
        dataset_name=dataset_name,
        input_config=config.input,
        cache_config=config.cache,
        logger=logger,
    )
    build_domains = functools.partial(
        dataset.build_domains,
        dataset_name=dataset_name,
        input_config=config.input,
        cache_config=config.cache,
        logger=logger
    )
    split_dataset = functools.partial(
        dataset.split_dataset,
        dataset_name=dataset_name,
        cache_config=config.cache,
        logger=logger
    )
    normalize_dataset = functools.partial(
        dataset.normalize_dataset,
        dataset_name=dataset_name,
        cache_config=config.cache,
        logger=logger
    )

    # build data blocks according to mode
    # training-only data
    if mode == 'train_only':
        build_data_cache(mode='training')
        build_domains(mode='training')
        split_dataset()
        normalize_dataset(mode='training')

    # inference-only data
    elif mode == 'infer_only':
        build_data_cache(mode='inference')
        build_domains(mode='inference')
        normalize_dataset(mode='inference')

    # both training and inference data
    elif mode == 'train_infer':
        build_data_cache(mode='training')
        build_data_cache(mode='inference')
        build_domains(mode='training')
        build_domains(mode='inference')
        split_dataset()
        normalize_dataset(mode='training')
        normalize_dataset(mode='inference')

    # catch wrong mode argument
    else:
        raise ValueError(
            f'Unsupported mode: {mode}. Should be one of the following:'
            f'"skip", "train_only", "infer_only", or "train_infer".'
        )

    # run data summary and return
    summary = dataset.summary.generate(dataset_name, config.cache)
    return summary
