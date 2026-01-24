'''Incoorperating domain knowledge rasters.'''

# standard imports
import dataclasses
import os
# third-party imports
import numpy
import rasterio
import rasterio.io
# local imports
import dataset.blocks
import utils

@dataclasses.dataclass
class DomainRasterBlock:
    '''doc'''
    name: str
    array: numpy.ndarray
    nodata: int
    mmin: int  = dataclasses.field(init=False)
    mmax: int  = dataclasses.field(init=False)

def parse(
        scheme_fpath: str,
        valid_fpath: str,
        domain_fpath: str,
        domain_config: list[dict] | None,
        *,
        logger: utils.Logger,
        overwrite: bool
    ) -> list[dict] | None:
    '''Inspect domain knowledge config and take actions accordingly.'''

    # get a child logger
    logger=logger.get_child('domkw')

    # check if already generated
    if os.path.exists(domain_fpath) and not overwrite:
        logger.log('INFO', f'Existing domain knowledge at: {domain_fpath}')
        return utils.load_json(domain_fpath)

    # get valid blocks and block window scheme
    blks = utils.load_pickle(valid_fpath)
    scheme = utils.load_pickle(scheme_fpath)

    # setup
    treated: list[dict] = []
    domains: list[dict] = []

    # iterate through provided domain rasters
    if domain_config is None:
        logger.log('INFO', 'No domain knowledge provided')
        return None
    for item in domain_config:
        # list of domain raster blocks
        bb = _parse_raster_blocks(blks, scheme, item['path'])
        # domain assignment according to configured treatment
        if item['treat'] == 'majority':
            treated = _majority(bb, item['name'])
            domains.append({dom['block_name']: dom for dom in treated})
            logger.log('INFO', f"Domain {item['name']} added from {item['path']} "
                            f'as the majority class at each block')
        if item['treat'] == 'pca':
            treated, var_exp = _pca_vectorize(bb, item['axes'], item['name'])
            domains.append({dom['block_name']: dom for dom in treated})
            logger.log('INFO', f"Domain {item['name']} added from {item['path']} "
                            f"as the first {item['axes']} PCA axes explaining "
                            f"{var_exp:.2f}% of the total variance")

    # merge domains and write to csv
    merged = _merge_domain(domains)
    utils.write_json(domain_fpath, merged)
    logger.log('INFO', f'Domain knowledge saved to: {domain_fpath}')
    return merged

def _parse_raster_blocks(
        blks: list[str],
        scheme: dict[str, rasterio.io.DatasetReader],
        domain_fpath: str,
    ) -> list[DomainRasterBlock]:
    '''Parse raster blocks via parallel processing.'''

    # get nodata of the domain
    nodata = _get_nodata(domain_fpath)

    # read through all raster blocks
    jobs = [(_read_ras_window, (b, scheme, domain_fpath), {}) for b in blks]
    results = utils.ParallelExecutor().run(jobs)

    # fetch results from reading the blocks
    parsed_blocks: list[DomainRasterBlock] = []
    unique_values = set()
    for r in results:
        name, arr, uniques = r
        parsed_blocks.append(DomainRasterBlock(name, arr, nodata))
        unique_values.update(uniques)
    if nodata in unique_values:
        unique_values.remove(nodata) # remove nodata if present

    # global mapping: raw 1..K  ->  0..K-1 ----
    # If global raws are continuous 1..K, K is simply max(unique_values)
    # but safer to compute via the set to allow future gaps.
    remap_sorted = numpy.array(sorted(unique_values), dtype=numpy.int64)
    kmax = remap_sorted.size
    if kmax == 0:
        # no valid class pixels at all; keep arrays as-is, but define a trivial space
        for b in parsed_blocks:
            b.mmin, b.mmax = 0, -1  # empty category space
        return parsed_blocks

    # Map each block: valid raw -> index in [0..K-1], nodata -> -1
    for b in parsed_blocks:
        arr = b.array
        mask_valid = arr != nodata
        # Initialize with -1 (nodata)
        mapped = numpy.full_like(arr, fill_value=-1, dtype=numpy.int64)
        # searchsorted assumes remap_sorted is sorted (it is)
        mapped[mask_valid] = numpy.searchsorted(remap_sorted, arr[mask_valid])
        b.array = mapped  # 0-based indices
        b.nodata = -1 # nodata is now -1
        # global index space for downstream consumers (fixed)
        b.mmin = 0
        b.mmax = kmax - 1

    # return
    return parsed_blocks

def _get_nodata(ras_fpath: str) -> int:
    '''Get nodata of an integer raster.'''

    with rasterio.open(ras_fpath) as src:
        nodata = src.nodata
    if nodata is None:
        nodata = -1
    else:
        assert abs(nodata - round(nodata)) < 1e-9 # nodata is a round number
    return int(nodata)

def _read_ras_window(
        block_fpath: str,
        blockscheme: dict[str, rasterio.io.DatasetReader],
        domain_ras_fpath: str
    ) -> tuple[str, numpy.ndarray, numpy.ndarray]:
    '''Get the `rasterio.windows.Window` from a given block fpath'''

    # use block name to retrieve raster read window from scheme
    block_name = dataset.blocks.parse_block_name(block_fpath).name
    block_window = blockscheme[block_name]

    # read domain raster
    with rasterio.open(domain_ras_fpath, 'r') as src:
        arr = src.read(1, window=block_window) # [C, H, W] always 3 dims

    # return block name, raster as an array, and unique values
    return block_name, arr, numpy.unique(arr)

def _majority(
        blk_list: list[DomainRasterBlock],
        domain_name: str
    ) -> list[dict[str, str | int]]:
    '''Get the most frequent class from the array.'''

    results: list[dict[str, str | int]] = []
    for b in blk_list:
        arr = b.array[b.array !=  b.nodata]
        values, counts = numpy.unique(arr, return_counts=True)
        results.append({
            'block_name': b.name,
            domain_name: values[numpy.argmax(counts)].item() # serializable
        })
    return results

def _pca_vectorize(
        blk_list: list[DomainRasterBlock],
        out_axes: int,
        domain_name: str
    ) -> tuple[list[dict[str, str | list[float]]], float]:
    '''Perform PCA and output top axes'''

    results: list[dict[str, str | list[float]]] = []
    names: list[str] = []
    frequencies: list[numpy.ndarray] = []

    for blk in blk_list:
        freq = _normal_frequency(blk.array, (blk.mmin, blk.mmax), blk.nodata)
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

def _normal_frequency(
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
