'''Block typing classes'''

# standard imports
import typing

# ---------------------------Block level meta dict---------------------------
class BlockGeneralMeta(typing.TypedDict):
    '''Block-level general metadata.'''
    block_name: str
    block_shape: tuple[int, int]
    valid_pixel_ratio: dict[str, float]
    has_label: bool

class BlockLabelMeta(typing.TypedDict):
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

class BlockImageMeta(typing.TypedDict):
    '''Block-level image metadata.'''
    image_nodata: float
    dem_pad: int
    band_map: dict[str, int]
    spectral_indices_added: list[str]
    topo_metrics_added: list[str]
    block_image_stats: dict[str, dict[str, int | float]]

class BlockMetaDict(BlockGeneralMeta, BlockImageMeta, BlockLabelMeta):
    '''Simple composite for the shape of a block meta dictionary.'''
