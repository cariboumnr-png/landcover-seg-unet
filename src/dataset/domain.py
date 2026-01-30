'''Incoorperating domain knowledge rasters.'''

from __future__ import annotations
# standard imports
import dataclasses
import os
import typing
# third-party imports
import numpy
import rasterio
# local imports
import _types
import utils

class DomainBlocks:
    '''doc'''

    def __init__(
            self,
            domain_name: str,
            domain_fpath: str,
            domain_config: _types.ConfigType,
        ):
        '''doc'''

        #
        self.domain_name = domain_name
        self.domain_fpath = domain_fpath
        self.domain_config = domain_config['domain_config'][domain_name]
        #
        self.parsed_blks: list[_DomainRasterBlock] = []
        self.domain_list: list[dict[str, typing.Any]] = []

    def process(
            self,
            valid_blks: dict[str, str],
            blks_layout: dict[str, _types.RasterWindow],
        ) -> list[str]:
        '''doc'''

        self.parse_raster_blocks(valid_blks, blks_layout)
        proc = _DomainConversion(self.parsed_blks)
        cfg = self.domain_config
        _name = self.domain_name

        if cfg.get('treat') == 'majority':
            self.domain_list = proc.majority(_name)
            return [
                f'Domain {_name} added from {self.domain_fpath}',
                'Treatment: Majority class in block'
            ]
        if cfg.get('treat') == 'pca':
            assert isinstance(cfg['axes'], int), \
                'PCA treatment requires specifying number of axes'
            axes = cfg['axes']
            self.domain_list, var_exp = proc.pca_vectorize(axes, _name)
            return [
                f'Domain {_name} added from {self.domain_fpath}',
                f'Treatment: PCA vectorized with the first {axes} axes',
                f'Total variance explained: {var_exp:.2f}%'
            ]
        raise ValueError(f'Unsupported treatment {cfg.get("treat")}')

    def parse_raster_blocks(
            self,
            valid_blks: dict[str, str],
            blks_layout: dict[str, _types.RasterWindow],
        ) -> None:
        '''Parse raster blocks via parallel processing.'''

        # read through all raster blocks
        jobs = [
            (self._read_window, (blk, blks_layout, self.domain_fpath), {})
            for blk in valid_blks
        ]
        results = utils.ParallelExecutor().run(jobs)

        # fetch results from reading the blocks
        unique_values = set()
        for (name, arr) in results:
            self.parsed_blks.append(_DomainRasterBlock(name, arr, self.nodata))
            unique_values.update(numpy.unique(arr))
        if self.nodata in unique_values:
            unique_values.remove(self.nodata) # remove nodata if present

        # global mapping: raw 1..K  ->  0..K-1 ----
        # If global raws are continuous 1..K, K is simply max(unique_values)
        # but safer to compute via the set to allow future gaps.
        remap = numpy.array(sorted(unique_values), dtype=numpy.int64)
        kmax = remap.size
        if kmax == 0:
            # no valid class pixels at all; keep arrays as-is,
            # but define a trivial space
            for b in self.parsed_blks:
                b.mmin, b.mmax = 0, -1  # empty category space
            return

        # Map each block: valid raw -> index in [0..K-1], nodata -> -1
        for b in self.parsed_blks:
            arr = b.array
            mask_valid = arr != self.nodata
            # Initialize with -1 (nodata)
            mapped = numpy.full_like(arr, fill_value=-1, dtype=numpy.int64)
            # searchsorted assumes remap_sorted is sorted (it is)
            mapped[mask_valid] = numpy.searchsorted(remap, arr[mask_valid])
            b.array = mapped  # 0-based indices
            b.nodata = -1 # nodata is now -1
            # global index space for downstream consumers (fixed)
            b.mmin = 0
            b.mmax = kmax - 1

    @staticmethod
    def _read_window(
            window_name: str,
            window_dict: dict[str, _types.RasterWindow],
            raster_fpath: str
        ) -> tuple[str, numpy.ndarray]:
        '''Read raster from a raster via a window.'''

        # use block name to retrieve raster read window from scheme
        block_window = window_dict[window_name]

        # read raster at window
        with rasterio.open(raster_fpath, 'r') as src:
            arr = src.read(1, window=block_window) # [C, H, W] always 3 dims

        # return array with name
        return window_name, arr

    @property
    def nodata(self) -> int:
        '''Nodata of the domain raster.'''
        with rasterio.open(self.domain_fpath) as src:
            nodata = src.nodata
        if nodata is None:
            nodata = -1
        else:
            assert abs(nodata - round(nodata)) < 1e-9 # nodata is a round number
        return int(nodata)

class _DomainConversion:
    '''doc'''

    def __init__(
            self,
            blk_list: list[_DomainRasterBlock]
        ):
        '''doc'''

        self.blk_list = blk_list

    def majority(
            self,
            domain_name: str
        ) -> list[dict[str, str | int]]:
        '''Get the most frequent class from the array.'''

        results: list[dict[str, str | int]] = []
        for b in self.blk_list:
            arr = b.array[b.array !=  b.nodata]
            values, counts = numpy.unique(arr, return_counts=True)
            results.append({
                'block_name': b.name,
                domain_name: values[numpy.argmax(counts)].item() # serializable
            })
        return results

    def pca_vectorize(
            self,
            out_axes: int,
            domain_name: str
        ) -> tuple[list[dict[str, str | list[float]]], float]:
        '''Perform PCA and output top axes'''

        results: list[dict[str, str | list[float]]] = []
        names: list[str] = []
        frequencies: list[numpy.ndarray] = []

        for blk in self.blk_list:
            freq = self._norm_freq(blk.array, (blk.mmin, blk.mmax), blk.nodata)
            names.append(blk.name)
            frequencies.append(freq)
        #
        freq_stack = numpy.stack(frequencies)
        pca_arr, var_explained = utils.pca_transform(freq_stack, out_axes)
        # iterate rows
        for i, row in enumerate(pca_arr):
            results.append({
                'block_name': names[i],
                domain_name: [float(x) for x in row]
            })

        return results, float(var_explained * 100)

    @staticmethod
    def _norm_freq(
            ras_array: numpy.ndarray,
            index_range: tuple[int, int],
            nodata_index: int
        ) -> numpy.ndarray:
        '''Get a normalized class frequency vector from the array.'''

        # remove nodata values
        valid = ras_array[ras_array != nodata_index]
        # sanity if array is all invalid
        if valid.size == 0:
            return numpy.zeros(index_range[1] - index_range[0] + 1)
        # get frequencies of valid elements
        values, counts = numpy.unique(valid, return_counts=True)
        frequencies = counts / counts.sum()
        # map class value to frequency
        freq_map = dict(zip(values, frequencies))
        # 1-based, inclusive
        i, j = index_range
        # return
        return numpy.array([freq_map.get(idx, 0.0) for idx in range(i, j + 1)])

@dataclasses.dataclass
class _DomainRasterBlock:
    '''doc'''
    name: str
    array: numpy.ndarray
    nodata: int
    mmin: int  = dataclasses.field(init=False)
    mmax: int  = dataclasses.field(init=False)

@dataclasses.dataclass
class _DomainProcessingContext:
    '''doc'''
    disc_domains: list[dict]
    cont_domains: list[dict]
    blks_dict: dict[str, str]
    layout_dict: dict[str, _types.RasterWindow]
    cfg: _types.ConfigType

def build_domains(
        dataset_name: str,
        input_config: _types.ConfigType,
        cache_config: _types.ConfigType,
        logger: utils.Logger,
        mode: str,
    ):
    '''Inspect domain knowledge config and take actions accordingly.'''

    # get a child logger
    logger=logger.get_child('domkw')

    # cache root directory
    cache_dir = f'./data/{dataset_name}/cache'

    # config accessors
    input_cfg = utils.ConfigAccess(input_config)
    cache_cfg = utils.ConfigAccess(cache_config)

    # get artifacts names
    blks_layout = cache_cfg.get_asset('artifacts', 'blocks', 'layout_dict')
    if mode == 'training':
        _blks = cache_cfg.get_asset('artifacts', 'blocks', 'valid')
    elif mode == 'inference':
        _blks = cache_cfg.get_asset('artifacts', 'blocks', 'square')
    else:
        raise ValueError('Mode must be either "training" or "inference".')
    domain_dict = cache_cfg.get_asset('artifacts', 'domain', 'by_block')

    # build according to mode
    logger.log('INFO', f'Adding domain to {mode} blocks')
    _build(
        contxt=_DomainProcessingContext(
            disc_domains=input_cfg.get_option('domain', 'disc') or [],
            cont_domains=input_cfg.get_option('domain', 'cont') or [],
            blks_dict=utils.load_json(f'{cache_dir}/{mode}/{_blks}'),
            layout_dict=utils.load_pickle(f'{cache_dir}/{mode}/{blks_layout}'),
            cfg=utils.load_json(input_cfg.get_option('config'))
        ),
        domain_save_fpath=f'{cache_dir}/{mode}/{domain_dict}',
        logger=logger,
        overwrite=cache_cfg.get_option('flags', 'overwrite_domain')
    )
    logger.log('INFO', 'Domain knowlage parsed')
    logger.log_sep()

def _build(
        contxt: _DomainProcessingContext,
        domain_save_fpath: str,
        logger: utils.Logger,
        overwrite: bool
    ):
    '''doc'''

    # check if already generated
    if os.path.exists(domain_save_fpath) and not overwrite:
        logger.log('INFO', f'Existing domain knowledge: {domain_save_fpath}')
        return utils.load_json(domain_save_fpath)

    # setup
    domains: list[dict] = []

    # iterate through provided domain rasters (discrete)
    logger.log('INFO', f'{len(contxt.disc_domains)} discrete domains provided')
    for dom in contxt.disc_domains:
        domain_blks = DomainBlocks(dom['name'], dom['path'], contxt.cfg)
        msgs = domain_blks.process(contxt.blks_dict, contxt.layout_dict)
        for m in msgs:
            logger.log('INFO', m)
        domains.append({d['block_name']: d for d in domain_blks.domain_list})

    # iterate through provided domain rasters (continuous)
    logger.log('INFO', f'{len(contxt.cont_domains)} continuous domains provided')
    for dom in contxt.cont_domains:
        domain_blks = DomainBlocks(dom['name'], dom['path'], contxt.cfg)
        msgs = domain_blks.process(contxt.blks_dict, contxt.layout_dict)
        for m in msgs:
            logger.log('INFO', m)
        domains.append({d['block_name']: d for d in domain_blks.domain_list})

    # merge domains and write to csv
    merged = _merge_domain(domains)
    utils.write_json(domain_save_fpath, merged)
    logger.log('INFO', f'Domain knowledge saved to: {domain_save_fpath}')
    return merged

def _merge_domain(list_domains: list[dict]) -> list[dict]:
    '''doc.'''

    merged = []
    # check if all domains have the same set of names
    name_lists = []
    for domain_blks in list_domains:
        name_lists.append(list(domain_blks.keys()))
    assert all(sorted(lst) == sorted(name_lists[0]) for lst in name_lists)

    #
    for blk_name in name_lists[0]:
        blk_domain = {'block_name': blk_name}
        for domain in list_domains:
            blk_domain.update(**domain[blk_name])
        merged.append(blk_domain)
    return merged
