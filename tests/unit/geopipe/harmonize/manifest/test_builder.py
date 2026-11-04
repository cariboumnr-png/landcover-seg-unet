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

'''Unit tests for dataset manifest builder (builder.py).'''

# standard imports
import pathlib
# third-party imports
import pytest
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.harmonize.manifest.builder as builder_mod


# ----- `DatasetManifestBuilder` construction tests
def test_builder_empty_compile_raises() -> None:
    '''
    Given: An empty builder with no registered raster entries.
    When: `compile()` is invoked.
    Then: Raise `DatasetManifestError`.
    '''
    builder = builder_mod.DatasetManifestBuilder()
    with pytest.raises(
        builder_mod.DatasetManifestError,
        match='Manifest has no raster entries',
    ):
        builder.compile()


def test_builder_add_feature_and_compile(tmp_path: pathlib.Path) -> None:
    '''
    Given: A multi-band feature raster configuration.
    When: `add_feature()` and `compile()` are invoked.
    Then: Compile canonical `ManifestEntry` with contiguous 1-based bands.
    '''
    tif_fp = tmp_path / 's2.tif'
    tif_fp.write_bytes(b'dummy')

    builder = builder_mod.DatasetManifestBuilder()
    builder.add_feature(
        name='sentinel2',
        path=tif_fp,
        bands=['blue', 'green', 'red', 'nir'],
        schemes={'rgb': ['blue', 'green', 'red']},
    )

    compiled = builder.compile()
    assert len(compiled) == 1
    entry = compiled[0]
    assert entry['name'] == 'sentinel2'
    assert entry['category'] == 'features'
    assert entry['band_mapping'] == {
        1: 'blue',
        2: 'green',
        3: 'red',
        4: 'nir',
    }
    assert entry['categorical_specs'] is None
    assert entry['schemes'] == {'rgb': ('blue', 'green', 'red')}


def test_builder_add_domain_and_compile(tmp_path: pathlib.Path) -> None:
    '''
    Given: A discrete domain raster configuration.
    When: `add_domain()` and `compile()` are invoked.
    Then: Compile canonical domain `ManifestEntry` with categorical specs.
    '''
    tif_fp = tmp_path / 'geology.tif'
    tif_fp.write_bytes(b'dummy')

    builder = builder_mod.DatasetManifestBuilder()
    builder.add_domain(
        name='geology',
        path=tif_fp,
        num_cls=4,
        index_base=1,
        ignore_cls=[255],
    )

    compiled = builder.compile()
    assert len(compiled) == 1
    entry = compiled[0]
    assert entry['name'] == 'geology'
    assert entry['category'] == 'domains'
    assert entry['schemes'] is None
    assert entry['categorical_specs'] == {
        'index_base': 1,
        'num_cls': 4,
        'ignore_cls': (255,),
    }


def test_builder_add_label_and_compile(tmp_path: pathlib.Path) -> None:
    '''
    Given: A categorical label raster configuration with reclass scheme.
    When: `add_label()` and `compile()` are invoked.
    Then: Compile canonical label `ManifestEntry` with class names and scheme.
    '''
    tif_fp = tmp_path / 'landcover.tif'
    tif_fp.write_bytes(b'dummy')

    builder = builder_mod.DatasetManifestBuilder()
    builder.add_label(
        name='landcover',
        path=tif_fp,
        num_cls=2,
        index_base=1,
        ignore_cls=[255],
        classes={1: 'coniferous', 2: 'deciduous'},
        color_map={1: (34, 139, 34), 2: (218, 165, 32)},
        schemes={
            'binary': {
                'reclass': {'1': [1, 2]},
                'reclass_name': {'1': 'forest'},
            }
        },
    )

    compiled = builder.compile()
    assert len(compiled) == 1
    entry = compiled[0]
    assert entry['name'] == 'landcover'
    assert entry['category'] == 'labels'
    schemes = entry['schemes']
    assert schemes is not None
    assert 'binary' in schemes


def test_builder_duplicate_name_raises(tmp_path: pathlib.Path) -> None:
    '''
    Given: Two layers registered with identical names.
    When: `compile()` is invoked.
    Then: Raise `DatasetManifestError` identifying duplicate name.
    '''
    tif_fp = tmp_path / 'img.tif'
    tif_fp.write_bytes(b'dummy')

    builder = builder_mod.DatasetManifestBuilder()
    builder.add_feature(name='img', path=tif_fp, bands=['b1'])
    builder.add_feature(name='img', path=tif_fp, bands=['b1'])

    with pytest.raises(
        builder_mod.DatasetManifestError,
        match='Duplicated raster name: img',
    ):
        builder.compile()


def test_builder_materialize_and_from_file_json(
    tmp_path: pathlib.Path,
) -> None:
    '''
    Given: Configured raster layers in builder.
    When: `materialize()` is called and reloaded via `from_file()` with JSON.
    Then: Successfully reload and compile matching manifest entries.
    '''
    tif_fp = tmp_path / 'feat.tif'
    tif_fp.write_bytes(b'dummy')

    builder = builder_mod.DatasetManifestBuilder()
    builder.add_feature(name='feat', path=tif_fp, bands=['b1', 'b2'])
    builder.materialize(tmp_path, name='manifest')

    json_fp = tmp_path / 'manifest.json'
    yaml_fp = tmp_path / 'manifest.yaml'
    assert json_fp.is_file()
    assert yaml_fp.is_file()

    reloaded = builder_mod.DatasetManifestBuilder.from_file(json_fp)
    compiled = reloaded.compile()
    assert len(compiled) == 1
    assert compiled[0]['name'] == 'feat'


def test_builder_from_file_yaml_and_dict_spec(
    tmp_path: pathlib.Path,
) -> None:
    '''
    Given: A single-document YAML manifest with relative path and 'rasters' key.
    When: `DatasetManifestBuilder.from_file()` is called on YAML.
    Then: Successfully parse, resolve relative paths, and compile entries.
    '''
    tif_fp = tmp_path / 'sample.tif'
    tif_fp.write_bytes(b'dummy')

    yaml_fp = tmp_path / 'dataset.yaml'
    spec = {
        'version': '1.0',
        'rasters': [
            {
                'name': 'sample',
                'path': 'sample.tif',
                'category': 'features',
                'band_mapping': {'1': 'b1'},
            }
        ],
    }
    artifacts.Controller(yaml_fp).persist(spec)

    builder = builder_mod.DatasetManifestBuilder.from_file(yaml_fp)
    compiled = builder.compile()
    assert len(compiled) == 1
    assert compiled[0]['name'] == 'sample'
    assert compiled[0]['path'].resolve() == tif_fp.resolve()

