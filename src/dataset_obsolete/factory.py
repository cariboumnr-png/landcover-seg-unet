'''Pipeline to prepare data from rasters for training.'''

# standard imports
import functools
# third-party imports
import omegaconf
# local imports
import dataset_obsolete
import utils

def prepare_data(
        dataset_name: str,
        config: omegaconf.DictConfig,
        logger: utils.Logger,
        *,
        mode: str
    ) -> dataset_obsolete.DataSummary:
    '''Data preparation pipeline.'''

    # if skip data preperation, e.g., continue training on existing data
    if mode == 'skip':
        # run data summary and return
        summary = dataset_obsolete.generate_summary(dataset_name, config.cache)
        return summary

    # otherwise prepare partially filed building functions
    build_data_cache = functools.partial(
        dataset_obsolete.build_data_cache,
        dataset_name=dataset_name,
        input_config=config.dataset,
        cache_config=config.cache,
        logger=logger,
    )
    build_domains = functools.partial(
        dataset_obsolete.build_domains,
        dataset_name=dataset_name,
        input_config=config.dataset,
        cache_config=config.cache,
        logger=logger
    )
    split_dataset = functools.partial(
        dataset_obsolete.split_dataset,
        dataset_name=dataset_name,
        cache_config=config.cache,
        logger=logger
    )
    normalize_dataset = functools.partial(
        dataset_obsolete.normalize_dataset,
        dataset_name=dataset_name,
        cache_config=config.cache,
        logger=logger
    )

    # build data blocks according to mode
    # both training and inference data
    if mode == 'with_inference':
        build_data_cache(mode='training')
        build_data_cache(mode='inference')
        build_domains(mode='training')
        build_domains(mode='inference')
        split_dataset()
        normalize_dataset(mode='training')
        normalize_dataset(mode='inference')

    # training-validation data
    elif mode == 'no_inference':
        build_data_cache(mode='training')
        build_domains(mode='training')
        split_dataset()
        normalize_dataset(mode='training')

    # inference-only data
    elif mode == 'inference_only':
        build_data_cache(mode='inference')
        build_domains(mode='inference')
        normalize_dataset(mode='inference')

    # catch wrong mode argument
    else:
        raise ValueError(
            f'Invalid mode: "{mode}". Must be in: ["skip", "with_inference",'
            f' "no_inference", or "inference_only"].'
        )

    # run data summary and return
    summary = dataset_obsolete.generate_summary(dataset_name, config.cache)
    return summary
