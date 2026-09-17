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

# pylint: disable=no-member

'''Unit tests for data harmonization processor (processor.py).'''

# standard imports
import dataclasses
import os
import typing
# third-party imports
import rasterio
# local imports
import landseg.geopipe.grid as grid
import landseg.geopipe.harmonize.manifest as manifest
import landseg.geopipe.harmonize.processor as processor
import landseg.geopipe.ingest.data_blocks.assembler as assembler


@dataclasses.dataclass
class _GridParams:
    ref_fpath: str
    crs_string: str = 'EPSG:3161'
    tile_size: tuple[int, int] = (16, 16)
    tile_stride: tuple[int, int] = (8, 8)
    origin: tuple[float, float] | None = None
    pixel_size: tuple[float, float] | None = None
    extent_in_crs_units: tuple[float, float] | None = None


# ----- `harmonize_sources` tests
def test_harmonize_sources_features_and_labels(
    tmp_path, dummy_geotiff_factory
):
    '''
    Given: Two feature rasters and one label raster.
    When: `harmonize_sources` is executed.
    Then: Warp rasters to grid and stack multiple features into one VRT.
    '''
    ref_path = dummy_geotiff_factory(
        filename='ref.tif', width=16, height=16, bands=1
    )
    s2_path = dummy_geotiff_factory(
        filename='s2.tif', width=16, height=16, bands=3
    )
    dem_path = dummy_geotiff_factory(
        filename='dem.tif', width=16, height=16, bands=1
    )
    lbl_path = dummy_geotiff_factory(
        filename='lbl.tif', width=16, height=16, bands=1
    )

    grid_params = _GridParams(ref_fpath=str(ref_path))
    world_grid = grid.build_grid('ref', grid_params)

    compiled: dict[str, manifest.ManifestEntry] = {
        str(s2_path): {
            'name': 's2',
            'path': str(s2_path),
            'category': 'features',
            'band_mapping': {1: 'blue', 2: 'green', 3: 'red'},
            'categorical_specs': None,
            'schemes': None,
        },
        str(dem_path): {
            'name': 'dem',
            'path': str(dem_path),
            'category': 'features',
            'band_mapping': {1: 'elevation'},
            'categorical_specs': None,
            'schemes': None,
        },
        str(lbl_path): {
            'name': 'landcover',
            'path': str(lbl_path),
            'category': 'labels',
            'band_mapping': {1: 'landcover'},
            'categorical_specs': {
                'index_base': 1,
                'num_cls': 2,
                'ignore_cls': [255],
            },
            'schemes': None,
        },
    }

    out_dir = str(tmp_path / 'harmonized')
    os.makedirs(out_dir, exist_ok=True)

    gen = processor.harmonize_sources(
        compiled,
        out_dir,
        world_grid,
        categorical_resampling='nearest',
        continuous_resampling='bilinear',
    )

    res: typing.Any = None
    try:
        while True:
            next(gen)
    except StopIteration as s:
        res = s.value

    assert isinstance(res, processor.ProcessedRasters)
    assert 'features' in res.finalized
    assert 'labels' in res.finalized
    assert res.finalized['features'].endswith(
        'harmonized_features_STACKED.vrt'
    )
    assert os.path.exists(res.finalized['features'])

    # verify stacked band count: 3 (s2) + 1 (dem) = 4
    with rasterio.open(res.finalized['features']) as src:
        assert src.count == 4


def test_harmonize_sources_domains(tmp_path, dummy_geotiff_factory):
    '''
    Given: A domain categorical raster.
    When: `harmonize_sources` is executed.
    Then: Fast-track domain raster into finalized without stacking.
    '''
    ref_path = dummy_geotiff_factory(
        filename='ref.tif', width=16, height=16, bands=1
    )
    dom_path = dummy_geotiff_factory(
        filename='eco.tif', width=16, height=16, bands=1
    )

    grid_params = _GridParams(ref_fpath=str(ref_path))
    world_grid = grid.build_grid('ref', grid_params)

    compiled: dict[str, manifest.ManifestEntry] = {
        str(dom_path): {
            'name': 'ecodistrict',
            'path': str(dom_path),
            'category': 'domains',
            'band_mapping': {1: 'ecodistrict'},
            'categorical_specs': None,
            'schemes': None,
        },
    }

    out_dir = str(tmp_path / 'harmonized')
    os.makedirs(out_dir, exist_ok=True)

    gen = processor.harmonize_sources(
        compiled,
        out_dir,
        world_grid,
        categorical_resampling='nearest',
        continuous_resampling='bilinear',
    )

    res: typing.Any = None
    try:
        while True:
            next(gen)
    except StopIteration as s:
        res = s.value

    assert isinstance(res, processor.ProcessedRasters)
    assert 'domains_ecodistrict' in res.finalized
    assert os.path.exists(res.finalized['domains_ecodistrict'])


def test_harmonize_sources_schemes_and_label_specs(
    tmp_path, dummy_geotiff_factory
):
    '''
    Given: Features and labels with sidecar schemes and
        categorical specs.
    When: Running `harmonize_sources`.
    Then: VRT tags propagate properly to downstream readers.
    '''
    s2_path = str(dummy_geotiff_factory(
        filename='s2.tif', width=16, height=16, bands=3
    ))
    lbl_path = str(dummy_geotiff_factory(
        filename='lbl.tif', width=16, height=16, bands=1
    ))
    world_grid = grid.build_grid(
        'ref',
        _GridParams(ref_fpath=str(dummy_geotiff_factory(
            filename='ref.tif', width=16, height=16, bands=1
        ))),
    )

    compiled: dict[str, manifest.ManifestEntry] = {
        s2_path: {
            'name': 's2',
            'path': s2_path,
            'category': 'features',
            'band_mapping': {1: 'blue', 2: 'green', 3: 'red'},
            'categorical_specs': None,
            'schemes': {'rgb': ['blue', 'green', 'red']},
        },
        lbl_path: {
            'name': 'landcover',
            'path': lbl_path,
            'category': 'labels',
            'band_mapping': {1: 'landcover'},
            'categorical_specs': {
                'index_base': 1,
                'num_cls': 2,
                'ignore_cls': [255],
                'class_name': {'1': 'forest', '2': 'water'},
            },
            'schemes': {
                'binary': {
                    'reclass': {'1': [1, 2]},
                    'reclass_name': {'1': 'veg'},
                }
            },
        },
    }

    os.makedirs(str(tmp_path / 'harmonized'), exist_ok=True)
    gen = processor.harmonize_sources(
        compiled,
        str(tmp_path / 'harmonized'),
        world_grid,
        categorical_resampling='nearest',
        continuous_resampling='bilinear',
    )

    res: typing.Any = None
    try:
        while True:
            next(gen)
    except StopIteration as s:
        res = s.value

    # verify single-label specs are readable downstream
    specs = assembler.read_label_specs(res.finalized['labels'])
    assert 'landcover' in specs
    assert specs['landcover']['num_cls'] == 2
    assert specs['landcover']['ignore_cls'] == [255]

    # verify schemes have dataset-level namespacing
    feat_schemes = assembler.read_schemes(res.finalized['features'])
    lbl_schemes = assembler.read_schemes(res.finalized['labels'])
    assert 's2' in feat_schemes
    assert feat_schemes['s2']['rgb'] == ['blue', 'green', 'red']
    assert 'landcover' in lbl_schemes
    assert 'binary' in lbl_schemes['landcover']
