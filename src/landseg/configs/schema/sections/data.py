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

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

'''
Data config schema
'''

# standard imports
import dataclasses
import re
import typing
# local imports
import landseg.configs.schema.utils as utils

# alias
field = dataclasses.field


# ----- world grid
@dataclasses.dataclass
class _GridParameters:
    tile_size: tuple[int, int] = (256, 256)
    tile_stride: tuple[int, int] = (128, 128)
    ref_fpath: str | None = None
    crs_string: str | None = None
    origin: tuple[float, float] | None = None
    pixel_size: tuple[float, float] | None = None
    extent_in_crs_units: tuple[float, float] | None = None

    def validate(self):
        # currently we only accept equal row and col sizes and strides
        if self.tile_size[0] != self.tile_size[1]:
            raise ValueError('Only square blocks are supported.')
        if self.tile_size[0] <= 0:
            raise ValueError('Block size must be positive.')

        if self.tile_stride[0] != self.tile_stride[1]:
            raise ValueError('Only equal row/column stride is supported.')
        if self.tile_stride[0] < 0:
            raise ValueError('Block stride must be zero or positive.')


@dataclasses.dataclass
class _GridCfg:
    mode: str = 'ref'
    params: _GridParameters = field(default_factory=_GridParameters)
    output_dpath: str = 'experiment/artifacts/world_grids'

    @property
    def tile_specs_tuple(self) -> tuple[int, int, int, int]:
        '''Tile specs in px as (row, col, overlap_row, overlap_col).'''
        return (
            self.params.tile_size[0],
            self.params.tile_size[1],
            self.params.tile_stride[0],
            self.params.tile_stride[1]
        )

    @property
    def spatial_resolution(self) -> float | None:
        '''Return resolution (CRS units per pixel). Assume square px.'''
        if self.params.pixel_size:
            return self.params.pixel_size[0]
        return None

    def validate(self) -> None:
        self.params.validate() # validates tile size and stride values

        if self.mode == 'ref':
            utils.must_exist(self.params.ref_fpath, 'grid reference raster')

        elif self.mode == 'manual':
            crs = self.params.crs_string
            if not crs or not bool(re.fullmatch(r'epsg:\d+', crs, re.I)):
                raise ValueError(f'Invalid CRS, must be [EPSG:....], got {crs}')

            if not self.params.origin:
                raise ValueError('Origin not provided')
            if not self.params.pixel_size:
                raise ValueError('Pixel size not provided')
            if not self.params.extent_in_crs_units:
                raise ValueError('Extent (in CRS units) not provided')

            if self.params.pixel_size[0] != self.params.pixel_size[1]:
                raise ValueError('Only square pixels are supported')


# ----- data harmonization
@dataclasses.dataclass
class _HarmonizationCfg:
    dataset_manifest: str = ''
    resampling_continuous: str = 'bilinear'
    resampling_categorical: str = 'nearest'
    output_dpath: str = 'experiment/artifacts/harmonized_data'

    def validate(self) -> None:
        pass


# ----- data ingestion
@dataclasses.dataclass
class _Domains:
    valid_threshold: float = 0.7
    target_variance: float = 0.9

    def validate(self) -> None:
        pass


@dataclasses.dataclass
class _DataBlocks:
    ignore_index: int = 255
    image_dem_pad: int = 8
    add_topo: list[str] | None = None
    add_spectral: list[str] | None = None

    def validate(self) -> None:
        if self.add_topo is not None:
            if not isinstance(self.add_topo, (list, tuple)):
                raise TypeError(
                    f'add_topo must be a list, '
                    f'got {type(self.add_topo)}'
                )
            for item in self.add_topo:
                if not isinstance(item, str):
                    raise TypeError(
                        f'Topo feature must be a string, got {type(item)}'
                    )
                if item.lower() not in ('slope', 'tpi'):
                    raise ValueError(
                        f'Invalid spectral index "{item}". '
                        'Supported featires are: slope, tpi.'
                    )

        if self.add_spectral is not None:
            if not isinstance(self.add_spectral, (list, tuple)):
                raise TypeError(
                    f'add_spectral must be a list, '
                    f'got {type(self.add_spectral)}'
                )
            for item in self.add_spectral:
                if not isinstance(item, str):
                    raise TypeError(
                        f'Spectral index must be a string, got {type(item)}'
                    )
                if item.lower() not in ('ndvi', 'ndmi', 'nbr'):
                    raise ValueError(
                        f'Invalid spectral index "{item}". '
                        'Supported indices are: ndvi, ndmi, nbr.'
                    )


@dataclasses.dataclass
class _IngestionCfg:
    domains: _Domains = field(default_factory=_Domains)
    datablocks: _DataBlocks = field(default_factory=_DataBlocks)
    rebuild: bool = False
    harmonization_run: int | str | None = None
    output_dpath: str = '${execution.exp_root}/artifacts/ingested_data'

    def validate(self) -> None:
        self.domains.validate()
        self.datablocks.validate()
        if self.harmonization_run is not None:
            if isinstance(self.harmonization_run, int):
                if self.harmonization_run <= 0:
                    raise ValueError(
                        'Harmonization run index must be positive.'
                    )
            elif isinstance(self.harmonization_run, str):
                if not self.harmonization_run.strip():
                    raise ValueError(
                        'Harmonization run identifier cannot be empty.'
                    )
            else:
                raise TypeError(
                    'Invalid harmonization_run type: '
                    f'{type(self.harmonization_run)}'
                )


# ----- data preparation
@dataclasses.dataclass
class _CatalogView:
    valid_pxs: dict[str, float] = field(default_factory=lambda: {'image': 0.9})
    focal_target: str | None = None
    test_catalog: str | None = None
    non_overlapping_test_grid: bool = True


    def validate(self):
        for k, v in self.valid_pxs.items():
            utils.must_within(v, f'{k} valid threshold', 0, 1)


@dataclasses.dataclass
class _Partition:
    val_ratio: float = 0.1
    test_ratio: float = 0.0
    buffer_step: int = 1
    train_aoi: str | None = None
    val_aoi: str | None = None
    test_aoi: str | None = None
    aoi_min_overlap: float = 0.5

    def validate(self):
        utils.must_within(self.val_ratio, 'validation block ratio', 0, 1)
        utils.must_within(self.test_ratio, 'test holdout block ratio', 0, 1)
        utils.must_within(self.aoi_min_overlap, 'AOI minimum overlap ratio', 0, 1)


@dataclasses.dataclass
class _Scoring:
    reward: dict[int, float] = field(default_factory=dict)
    alpha: float = 1.0
    beta: float = 0.0

    def validate(self):
        utils.must_within(self.alpha, 'scoring alpha', 0)
        utils.must_within(self.beta, 'scoring beta', 0)


@dataclasses.dataclass
class _Hydration:
    max_skew_rate: float = 10.0

    def validate(self):
        utils.must_within(self.max_skew_rate, 'hydration skew ratio', 0)


@dataclasses.dataclass
class _PreparationCfg:
    features: dict[str, typing.Any] = field(default_factory=dict)
    targets: dict[str, typing.Any] = field(default_factory=dict)
    catalog: _CatalogView = field(default_factory=_CatalogView)
    partition: _Partition = field(default_factory=_Partition)
    scoring: _Scoring = field(default_factory=_Scoring)
    hydration: _Hydration = field(default_factory=_Hydration)
    rebuild: bool = False
    output_dpath: str = '${execution.exp_root}/artifacts/prepared_data'

    def validate(self):
        self.catalog.validate()
        self.partition.validate()
        self.scoring.validate()
        self.hydration.validate()

        for k, v in self.features.items():
            if not isinstance(k, str) or not isinstance(v, (str, list, bool)):
                raise ValueError(
                    f'Invalid features config for "{k}": '
                    f'expected string, list of strings, or boolean'
                )

        for k, v in self.targets.items():
            if not isinstance(k, str) or not isinstance(v, (str, dict)):
                raise ValueError(
                    f'Invalid targets config for "{k}": '
                    f'expected string or dict'
                )


# ----- data specs
@dataclasses.dataclass
class _Specification:
    domain_ids_name: str | None = None
    domain_vec_name: str | None = None


# ----- composite
@dataclasses.dataclass
class DataConfig:
    world_grid: _GridCfg = field(default_factory=_GridCfg)
    harmonization: _HarmonizationCfg = field(default_factory=_HarmonizationCfg)
    ingestion: _IngestionCfg = field(default_factory=_IngestionCfg)
    preparation: _PreparationCfg = field(default_factory=_PreparationCfg)
    specification: _Specification = field(default_factory=_Specification)

    def validate(self):
        self.world_grid.validate()
        self.harmonization.validate()
        self.ingestion.validate()
        self.preparation.validate()
