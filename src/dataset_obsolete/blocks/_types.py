'''Typed dictionaries used in this module.'''

# standard imports
import typing

# ----- Block Meta dict
class _GeneralMeta(typing.TypedDict):
    '''Block-level general metadata.'''
    block_name: str
    block_shape: tuple[int, int]
    valid_pixel_ratio: dict[str, float]
    has_label: bool

class _LabelMeta(typing.TypedDict):
    '''Block-level label metadata.'''
    label_nodata: int
    ignore_label: int
    label1_num_classes: int
    label1_to_ignore: list[int]
    label1_class_name: dict[str, str]
    label1_reclass_map: dict[str, list[int]]
    label1_reclass_name: dict[str, str]
    label_count: dict[str, list[int]]
    label_entropy: dict[str, float]

class _ImageMeta(typing.TypedDict):
    '''Block-level image metadata.'''
    image_nodata: float
    dem_pad: int
    band_map: dict[str, int]
    spectral_indices_added: list[str]
    topo_metrics_added: list[str]
    block_image_stats: dict[str, dict[str, int | float]]

class BlockMeta(_GeneralMeta, _ImageMeta, _LabelMeta):
    '''Defines the shape of a block meta dictionary.'''

# ------ Global image stats dict
class ImageStats(typing.TypedDict):
    '''Block-level image stats used in Welford's Online Algorithm.'''
    total_count: int
    current_mean: float
    accum_m2: float
    std: float
