# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      © King's Printer for Ontario, 2026.                    #
#                                                                             #
#       Licensed under the Apache License, Version 2.0 (the 'License');       #
#          you may not use this file except in compliance with the            #
#                                  License.                                   #
#                  You may obtain a copy of the License at:                   #
#                                                                             #
#                  http://www.apache.org/licenses/LICENSE-2.0                 #
#                                                                             #
#    Unless required by applicable law or agreed to in writing, software      #
#     distributed under the License is distributed on an 'AS IS' BASIS,       #
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        #
#                                   implied.                                  #
#       See the License for the specific language governing permissions       #
#                       and limitations under the License.                    #
# =========================================================================== #

'''
Container and serialization interface for geospatial data blocks.

Defines the core storage container and compressed `.npz` archive serialization
contracts for windowed raster arrays and structured manifest metadata.

Public APIs:
    - DataBlockManifest: typed dictionary defining block manifest metadata.
    - DataBlockArrays: container for block-wise image and label arrays.
    - DataBlock: container for raster block arrays and manifest metadata.
'''

# standard imports
from __future__ import annotations
import dataclasses
import json
import typing
# third-party imports
import numpy
# local imports
import landseg.geopipe.core as geo_core


# ----- public types
class DataBlockManifest(typing.TypedDict):
    '''
    Typed dictionary defining the persisted manifest for a data block.

    The manifest records block provenance, dataset schema, and derived
    statistics required to interpret the serialized arrays independently
    of the original dataset configuration.
    '''
    # provenance
    block_name: str
    has_label: bool
    # image description
    image_band_map: dict[str, int]
    image_nodata: float
    # label descriptions
    label_band_map: dict[str, int] # head name: index
    label_nodata: int
    label_ignore_index: int # ignore_idx to convert to globally
    label_ignore_cls: dict[str, list[int]] # head name: list of ignores
    label_num_cls: dict[str, int]
    label_cls_names: dict[str, list[str]] # head name: list of class names
    label_cls_clr_map: dict[str, dict[str, list[int]]] # {head: {class: RGB}}
    label_taxonomy: dict[str, geo_core.TaxonomySpecs]

    # derived stats
    valid_ratios: dict[str, float]
    image_stats: dict[str, dict[str, int | float]]
    label_count: dict[str, list[int]]
    label_entropy: dict[str, float]


# ----- public dataclasses
@dataclasses.dataclass
class DataBlockArrays:
    '''Container for block-wise image, label, and valid mask arrays.'''
    image: numpy.ndarray
    label: numpy.ndarray
    valid_mask: numpy.ndarray

    def validate(self) -> None:
        '''Validate presence and shape consistency of arrays.'''
        if self.image.ndim != 3:
            raise ValueError('Image array is not of shape [C, H, W]')
        if self.valid_mask.ndim != 2:
            raise ValueError('Valid mask array is not of shape [H, W]')
        if self.image.shape[-2:] != self.valid_mask.shape:
            raise ValueError(
                'Image and valid mask arrays have mismatched H / W'
            )
        if self.label.ndim != 3 and self.label.shape != (1,):
            raise ValueError('Label array is not of shape [C, H, W] or dummy')


# ----- public classes
class DataBlock:
    '''
    Container for per-block raster data and its associated manifest.

    Encapsulates a single raster window along with its associated
    manifest and serialized arrays. Supports loading from and persisting to
    compressed `.npz` artifacts.

    Typical workflow:
        - Use `load()` to restore a previously saved block from disk
        - Access `.data.image`, `.data.label`, and `.data.valid_mask`
        - Access `.manifest` for block provenance and channel statistics
        - Use `save()` to persist the block to disk
    '''

    def __init__(
        self,
        data: DataBlockArrays | None = None,
        manifest: DataBlockManifest | None = None,
    ):
        '''Initialize a DataBlock with arrays and manifest metadata.'''
        if data is None:
            self.data = DataBlockArrays(
                image=numpy.array([], dtype=numpy.float32),
                label=numpy.array([1]),
                valid_mask=numpy.array([], dtype=bool),
            )
        else:
            self.data = data

        self.manifest = (
            manifest if manifest is not None else self.empty_manifest()
        )

    @classmethod
    def load(cls, fpath: str) -> 'DataBlock':
        '''
        Load a DataBlock from a serialized .npz file.

        Reconstructs both the data arrays and manifest metadata from
        a previously saved block artifact.

        Args:
            fpath:
                path to the .npz file containing serialized data.

        Returns:
            DataBlock:
                populated block instance with restored arrays and
                manifest.
        '''
        loaded = numpy.load(fpath)
        manifest = json.loads(loaded['manifest_json'].item())
        arrays = DataBlockArrays(
            image=loaded['image'],
            label=loaded['label'],
            valid_mask=loaded['valid_mask'],
        )
        return cls(data=arrays, manifest=manifest)

    def save(self, fpath: str) -> None:
        '''
        Save the DataBlock to a compressed .npz file.

        Serializes all internal arrays and manifest dictionary as a
        compressed numpy archive artifact.

        Args:
            fpath:
                output file path ending with .npz.
        '''
        if not fpath.endswith('.npz'):
            raise ValueError(f'Output path must end with .npz: {fpath}')

        manifest_str = json.dumps(self.manifest, separators=(',', ':'))
        numpy.savez_compressed(
            fpath,
            image=self.data.image,
            label=self.data.label,
            valid_mask=self.data.valid_mask,
            manifest_json=manifest_str,
        )

    @staticmethod
    def empty_manifest(block_name: str = '') -> DataBlockManifest:
        '''Generate a default empty manifest dictionary.'''
        return {
            'block_name': block_name,
            'has_label': False,
            'image_band_map': {},
            'image_nodata': numpy.nan,
            'label_band_map': {},
            'label_nodata': 0,
            'label_ignore_index': 255,
            'label_ignore_cls': {},
            'label_num_cls': {},
            'label_cls_names': {},
            'label_cls_clr_map': {},
            'label_taxonomy': {},
            'valid_ratios': {},
            'image_stats': {},
            'label_count': {},
            'label_entropy': {},
        }
