# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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

# pylint: disable=missing-function-docstring

'''
Dummy dataset generator for landseg integration testing and local execution.
'''

# standard imports
import dataclasses
import json
import os
import typing
# third-party imports
import numpy
import rasterio
import rasterio.transform
import rasterio.warp


# ----- public dataclass
@dataclasses.dataclass
class TIFFConfig:
    '''Container for creating a dummy GeoTIFF file via `rasterio`.'''
    shape: tuple[int, int]
    bands: int
    crs: str
    transform: rasterio.transform.Affine
    dtype: typing.Any
    nodata: int | float


@dataclasses.dataclass
class TIFFPaths:
    '''Container for dummy TIFF file paths.'''
    root: str

    @property
    def extent(self) -> str:
        return self._make_path('reference_raster/sample_extent.tif')

    @property
    def test_aoi(self) -> str:
        return self._make_path('reference_raster/sample_test_aoi.tif')

    @property
    def domain_1(self) -> str:
        return self._make_path('raw_data/sample_domain_1.tif')

    @property
    def domain_2(self) -> str:
        return self._make_path('raw_data/sample_domain_2.tif')

    @property
    def sentinel2(self) -> str:
        return self._make_path('raw_data/sample_sentinel2.tif')

    @property
    def dem(self) -> str:
        return self._make_path('raw_data/sample_dem.tif')

    @property
    def landcover(self) -> str:
        return self._make_path('raw_data/sample_landcover.tif')

    @property
    def leadspc(self) -> str:
        return self._make_path('raw_data/sample_leadspc.tif')

    @property
    def manifest(self) -> str:
        return self._make_path('raw_data/manifest.json')

    @property
    def all_paths_exist(self) -> bool:
        return all(
            os.path.exists(p) for p in [
                self.extent,
                self.test_aoi,
                self.domain_1,
                self.domain_2,
                self.sentinel2,
                self.dem,
                self.landcover,
                self.leadspc,
                self.manifest,
            ]
        )

    def _make_path(self, p: str) -> str:
        return os.path.abspath(os.path.join(self.root, p))


# ----- public function
def create_dummy_geotiff(
    fpath: str,
    *,
    config: TIFFConfig,
    data_gen_func: typing.Callable[[tuple[int, int], int], numpy.ndarray],
) -> None:
    '''Write a multi-band GeoTIFF with coordinate metadata.

    Args:
        fpath:
            Output file path.
        config:
            TIFFConfig with shape, bands, crs, transform, dtype, nodata.
        data_gen_func:
            Callback generating data per band.
    '''
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with rasterio.open(
        fpath,
        'w',
        driver='GTiff',
        height=config.shape[0],
        width=config.shape[1],
        count=config.bands,
        dtype=config.dtype,
        crs=config.crs,
        transform=config.transform,
        nodata=config.nodata,
    ) as dst:
        for b in range(1, config.bands + 1):
            band_data = data_gen_func(config.shape, b)
            dst.write(band_data, b)


def generate_dummy_data(input_root: str = './experiment/input') -> TIFFPaths:
    '''Generate the full dummy dataset under input root.

    Args:
        input_root:
            Root directory for output files.
    '''
    print('Generating dummy geospatial data for landseg pipeline...')
    os.makedirs(input_root, exist_ok=True)
    paths = TIFFPaths(input_root)

    canonical_crs = 'EPSG:3161'
    # global transform (canonical CRS)
    global_transform = _get_transform(500000.0, 5000000.0, 20)
    # 512 height x 768 width gives 2 rows x 3 cols = 6 base blocks (20m res)
    global_shape = (512, 768)

    # -----create extent reference
    # shape: global
    # transform: global
    print(f'Creating extent reference: {paths.extent}')
    create_dummy_geotiff(
        paths.extent,
        config=TIFFConfig(
            shape=global_shape,
            bands=1,
            crs=canonical_crs,
            transform=global_transform,
            dtype=numpy.uint8,
            nodata=0,
        ),
        data_gen_func=lambda s, b: numpy.ones(s, dtype=numpy.uint8),
    )

    # ----- create test AOI raster
    # shape: (256x256)
    # transform: global (so it covers exactly the top-left block)
    print(f'Creating test AOI reference: {paths.test_aoi}')
    create_dummy_geotiff(
        paths.test_aoi,
        config=TIFFConfig(
            shape=(256, 256),
            bands=1,
            crs=canonical_crs,
            transform=global_transform,
            dtype=numpy.uint8,
            nodata=0,
        ),
        data_gen_func=lambda s, b: numpy.ones(s, dtype=numpy.uint8),
    )

    # each data raster adds a config JSON dict
    data_configs: list[dict[str, typing.Any]] = []

    # ----- create domains rasters
    # shape: global
    # transform: global
    print(f'Creating domain knowledge [mock_geology]: {paths.domain_1}')
    create_dummy_geotiff(
        paths.domain_1,
        config=TIFFConfig(
            shape=global_shape,
            bands=1,
            crs=canonical_crs,
            transform=global_transform,
            dtype=numpy.uint8,
            nodata=255,
        ),
        data_gen_func=lambda s, b: numpy.random.randint(
            1, 5, size=s, dtype=numpy.uint8
        ),
    )
    data_configs.append({
        'name': 'mock_geology',
        'path': paths.domain_1,
        'category': 'domains',
        'band_mapping': {
            1: "mock_geology"
        },
        'categorical_specs': {
            'index_base': 1,
            'num_cls': 4,
            'ignore_cls': [255],
        },
        'schemes': None,
    })

    print(f'Creating domain knowledge [mock_soil]: {paths.domain_2}')
    create_dummy_geotiff(
        paths.domain_2,
        config=TIFFConfig(
            shape=global_shape,
            bands=1,
            crs=canonical_crs,
            transform=global_transform,
            dtype=numpy.uint8,
            nodata=255,
        ),
        data_gen_func=lambda s, b: numpy.random.randint(
            1, 10, size=s, dtype=numpy.uint8
        ),
    )
    data_configs.append({
        'name': 'mock_soil',
        'path': paths.domain_2,
        'category': 'domains',
        'band_mapping': {
            1: "mock_soil"
        },
        'categorical_specs': {
            'index_base': 1,
            'num_cls': 9,
            'ignore_cls': [255],
        },
        'schemes': None,
    })

    # ----- create feature and label rasters
    # shape: double dimensions in 10m resolution
    # transform: different UTM CRS (to test harmonization)
    raw_shape = (512 * 2, 768 * 2)
    raw_crs = 'EPSG:2958'
    raw_origin = rasterio.warp.transform(
        canonical_crs,
        raw_crs,
        [500000.0],
        [5000000.0]
    )
    raw_transform = _get_transform(raw_origin[0][0], raw_origin[1][0], 10.0)

    print(f'Creating raw feature [Sentinel-2]: {paths.sentinel2}')
    create_dummy_geotiff(
        paths.sentinel2,
        config=TIFFConfig(
            shape=raw_shape,
            bands=10,
            crs=raw_crs,
            transform=raw_transform,
            dtype=numpy.uint16,
            nodata=65535,
        ),
        data_gen_func=lambda s, b: numpy.random.randint(
            100, 3000, size=s, dtype=numpy.uint16
        ),
    )
    data_configs.append({
        'name': 'sentinel2',
        'path': paths.sentinel2,
        'category': 'features',
        'band_mapping': {
            1: 'blue',
            2: 'green',
            3: 'red',
            4: 'red_edge1',
            5: 'red_edge2',
            6: 'red_edge3',
            7: 'nir',
            8: 'narrow_nir',
            9: 'swir1',
            10: 'swir2',
        },
        'categorical_specs': None,
        'schemes': {
            'rgb': ['blue', 'green', 'red'],
            'rgb_nir': ['blue', 'green', 'red', 'nir'],
            'surface': [
                'blue', 'green', 'red', 'red_edge1', 'red_edge2',
                'red_edge3', 'nir', 'narrow_nir', 'swir1', 'swir2'
            ],
        },
    })

    print(f'Creating raw feature [DEM]: {paths.dem}')
    create_dummy_geotiff(
        paths.dem,
        config=TIFFConfig(
            shape=raw_shape,
            bands=1,
            crs=raw_crs,
            transform=raw_transform,
            dtype=numpy.float32,
            nodata=-9999.9,
        ),
        data_gen_func=lambda s, _: _gen_image_data(s, 1),
    )
    data_configs.append({
        'name': 'dem',
        'path': paths.dem,
        'category': 'features',
        'band_mapping': {
            1: 'dem',
        },
        'categorical_specs': None,
        'schemes': None,
    })

    print(f'Creating raw label [Landcover]: {paths.landcover}')
    create_dummy_geotiff(
        paths.landcover,
        config=TIFFConfig(
            shape=raw_shape,
            bands=1,
            crs=raw_crs,
            transform=raw_transform,
            dtype=numpy.uint8,
            nodata=255,
        ),
        data_gen_func=_gen_label_data,
    )
    data_configs.append({
        'name': 'landcover',
        'path': paths.landcover,
        'category': 'labels',
        'band_mapping': {
            1: 'landcover'
        },
        'categorical_specs': {
            'index_base': 1,
            'num_cls': 2,
            'ignore_cls': [255],
            'class_name': {
                '1': 'coniferous',
                '2': 'deciduous',
            },
            'color_map': {
                '1': [34, 139, 34],
                '2': [218, 165, 32],
            },
        },
        'schemes': {
            'binary': {
                'reclass': {'1': [1, 2]},
                'reclass_name': {'1': 'forest'},
            },
        },
    })

    print(f'Creating raw label [Leadspc]: {paths.leadspc}')
    create_dummy_geotiff(
        paths.leadspc,
        config=TIFFConfig(
            shape=raw_shape,
            bands=1,
            crs=raw_crs,
            transform=raw_transform,
            dtype=numpy.uint8,
            nodata=255,
        ),
        data_gen_func=_gen_label_data,
    )
    data_configs.append({
        'name': 'leadspc',
        'path': paths.leadspc,
        'category': 'labels',
        'band_mapping': {
            1: 'leadspc'
        },
        'categorical_specs': {
            'index_base': 1,
            'num_cls': 2,
            'ignore_cls': [255],
            'class_name': {
                '1': 'SB_BLACK_SPRUCE',
                '2': 'PJ_JACK_PINE',
            },
            'color_map': {
                '1': [0, 100, 0],
                '2': [107, 142, 35],
            },
            'taxonomy': {
                'profile': 'ontario_tree_species_grouped_profiles',
            },
        },
        'schemes': None,
    })

    _write_jsons(data_configs, paths.manifest)
    print('\nDummy data generation completed successfully!')
    return paths


# ----- private functions
def _get_transform(x: float, y: float, res: float) -> rasterio.transform.Affine:
    '''Get transform from origin and resolution.'''
    return rasterio.transform.from_origin(x, y, res, res)


def _write_jsons(src: list[dict[str, typing.Any]], manifest: str) -> None:
    '''Write per file sidecar JSON and manifest JSON.'''
    # clear existing hash record if regenerating
    hash_fpath = os.path.join(os.path.dirname(manifest), '_hash.json')
    if os.path.exists(hash_fpath):
        os.remove(hash_fpath)

    manifest_list: list[dict[str, typing.Any]] = []
    for cfg in src:
        json_fpath = cfg['path'].replace('tif', 'json')
        with open(json_fpath, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=4)
        manifest_list.append({
            'name': cfg['name'],
            'path': cfg['path'],
            'manifest': json_fpath,
        })
    with open(manifest, 'w', encoding='utf-8') as f:
        json.dump(manifest_list, f, indent=4)


def _gen_image_data(shape: tuple[int, int], band_idx: int) -> numpy.ndarray:
    '''Generate dummy image band or terrain elevation data.'''
    if band_idx == 1:
        x = numpy.linspace(100.0, 300.0, shape[1])
        return numpy.tile(x, (shape[0], 1)).astype(numpy.float32)
    return numpy.random.randint(
        100, 1000, size=shape, dtype=numpy.uint16
    )


def _gen_label_data(shape: tuple[int, int], _: int) -> numpy.ndarray:
    '''Generate dummy label data with ignore index.'''
    labels = numpy.random.choice([1, 2, 255], size=shape, p=[0.45, 0.45, 0.10])
    return labels.astype(numpy.uint8)
