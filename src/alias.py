'''Project-wide type aliases and lazy imports for type checking.'''

# standard imports
import typing
# third-party imports
import numpy
import rasterio.io
import rasterio.windows
import torch

# typing aliases
# generic
ConfigType: typing.TypeAlias = typing.Mapping[str, typing.Any]
'''
Generic string-keyed config mapping, e.g., from omega dict.
'''
# batch context
Tensor: typing.TypeAlias = torch.Tensor
TorchDict: typing.TypeAlias = dict[str, Tensor]
DatasetItem: typing.TypeAlias = tuple[Tensor, Tensor, TorchDict]
'''
A tuple from one sample of the dataset: x (always present), y (can be
a placeholder during inference, e.g., `torch.Tensor([1])`) and domain
(always present but can be empty).
'''
DatasetBatch: typing.TypeAlias = typing.Sequence[DatasetItem]
'''
A collection of `DatasetItem` objects.
'''
# from rasterio
RasterReader: typing.TypeAlias = rasterio.io.DatasetReader
RasterWindow: typing.TypeAlias = rasterio.windows.Window
#
RasterTile: typing.TypeAlias = tuple[tuple[int, int], numpy.ndarray]
'''
Array read from a raster window with its top-left corner at specified
coordinates (x, y in pixels) from the world grid.
'''
