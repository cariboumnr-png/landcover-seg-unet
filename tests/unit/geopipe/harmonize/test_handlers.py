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

'''Unit tests for geopipe harmonize domain handlers (handlers.py).'''

# standard imports
import json
import pathlib
import typing
# third-party imports
import numpy
import pytest
import rasterio.transform
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.harmonize.handlers as handlers
import landseg.geopipe.harmonize.manifest.compiler as compiler
import landseg.testing.tiff as tiff
import landseg.utils.validator as val


# ----- `validate_manifest_contract` tests
def test_validate_manifest_contract_success(tmp_path):
    '''
    Given: A valid dataset manifest referencing feature and label entries.
    When: `validate_manifest_contract` is called.
    Then: Return a tuple of compiled manifest entries.
    '''
    s2_tif = _create_dummy_raster(tmp_path / 's2.tif', bands=3)
    s2_sub_fp = _write_sub_manifest(
        tmp_path / 's2_manifest.json',
        name='s2',
        path=s2_tif,
        category='features',
        band_mapping={1: 'blue', 2: 'green', 3: 'red'},
    )

    lc_tif = _create_dummy_raster(tmp_path / 'lc.tif', bands=1)
    lc_sub_fp = _write_sub_manifest(
        tmp_path / 'lc_manifest.json',
        name='lc',
        path=lc_tif,
        category='labels',
        band_mapping={1: 'landcover'},
        categorical_specs={
            'index_base': 1,
            'num_cls': 5,
            'ignore_cls': [255],
        },
    )

    root_fp = tmp_path / 'dataset_manifest.json'
    _write_root_manifest(
        root_fp,
        [
            {'name': 's2', 'path': s2_tif, 'manifest': s2_sub_fp},
            {'name': 'lc', 'path': lc_tif, 'manifest': lc_sub_fp},
        ],
    )

    entries = handlers.validate_manifest_contract(root_fp)
    assert len(entries) == 2
    assert entries[0]['name'] == 's2'
    assert entries[1]['name'] == 'lc'


def test_validate_manifest_contract_missing_file(tmp_path):
    '''
    Given: A non-existent manifest file path.
    When: `validate_manifest_contract` is called.
    Then: Raise an error indicating the file is missing.
    '''
    missing_fp = tmp_path / 'non_existent.json'
    with pytest.raises((FileNotFoundError, artifacts.ArtifactError)):
        handlers.validate_manifest_contract(missing_fp)


def test_validate_manifest_contract_compiler_failure(tmp_path):
    '''
    Given: A malformed root manifest JSON containing a dictionary.
    When: `validate_manifest_contract` is called.
    Then: Raise a DatasetManifestError.
    '''
    invalid_fp = tmp_path / 'malformed.json'
    with open(invalid_fp, 'w', encoding='UTF-8') as f:
        json.dump({'not': 'a list'}, f)

    with pytest.raises(
        (compiler.DatasetManifestError, val.ValidationError),
    ):
        handlers.validate_manifest_contract(invalid_fp)


# ----- `validate_grid_reference` tests
def test_validate_grid_reference_success(tmp_path):
    '''
    Given: A valid world grid artifact and corresponding report.
    When: `validate_grid_reference` is called.
    Then: Return restored GridLayout, grid_id, and grid_fpath.
    '''
    grid, report_fp, grid_fp = _create_dummy_grid_artifacts(tmp_path)

    restored_grid, grid_id, restored_fp = handlers.validate_grid_reference(
        report_fp
    )

    assert grid_id == grid.gid
    assert restored_fp == grid_fp
    assert len(restored_grid) == len(grid)


def test_validate_grid_reference_missing_report(tmp_path):
    '''
    Given: A non-existent grid report file path.
    When: `validate_grid_reference` is called.
    Then: Raise an ArtifactError or FileNotFoundError.
    '''
    missing_fp = tmp_path / 'missing_report.json'
    with pytest.raises((artifacts.ArtifactError, FileNotFoundError)):
        handlers.validate_grid_reference(missing_fp)


# ----- private helpers
def _create_dummy_raster(
    fpath: pathlib.Path,
    bands: int = 1,
    crs: str = 'EPSG:32617',
) -> pathlib.Path:
    '''Create a minimal valid GeoTIFF on disk using tiff utility.'''
    fpath.parent.mkdir(parents=True, exist_ok=True)
    transform = rasterio.transform.from_origin(0.0, 1000.0, 10.0, 10.0)
    tiff.create_dummy_geotiff(
        fpath,
        config=tiff.TIFFConfig(
            shape=(16, 16),
            bands=bands,
            crs=crs,
            transform=transform,
            dtype=numpy.uint8,
            nodata=0,
        ),
        data_gen_func=lambda s, b: numpy.ones(s, dtype=numpy.uint8),
    )
    return fpath


def _create_dummy_grid_artifacts(
    tmp_path: pathlib.Path,
) -> tuple[geo_core.GridLayout, pathlib.Path, pathlib.Path]:
    '''Persist a minimal GridLayout and GridReport to disk.'''
    spec = geo_core.GridSpec(
        crs='EPSG:32617',
        origin=(0.0, 1000.0),
        pixel_size=(10.0, 10.0),
        tile_size=(16, 16),
        tile_stride=(8, 8),
        grid_extent=(160.0, 160.0),
    )
    world_grid = geo_core.GridLayout(spec)
    grid_fpath = tmp_path / 'grid.json'
    grid_ctrl = artifacts.Controller[geo_core.GridPayload](
        grid_fpath,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
        schema_id=geo_core.GridLayout.SCHEMA_ID,
    )
    grid_ctrl.persist(world_grid.to_payload())

    report_fpath = tmp_path / 'grid_report.json'
    report_ctrl = artifacts.Controller[dict](
        report_fpath,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
    )
    report_ctrl.persist({
        'status': 'SUCCESS',
        'grid': {
            'grid_id': world_grid.gid,
            'grid_fpath': str(grid_fpath),
        },
    })
    return world_grid, report_fpath, grid_fpath


def _write_sub_manifest(
    fpath: pathlib.Path,
    name: str,
    path: pathlib.Path,
    category: str,
    band_mapping: dict[int, str],
    categorical_specs: dict | None = None,
) -> pathlib.Path:
    '''Write sub-manifest JSON file for an individual dataset entry.'''
    data = {
        'name': name,
        'path': str(path),
        'category': category,
        'band_mapping': {str(k): v for k, v in band_mapping.items()},
    }
    if categorical_specs:
        data['categorical_specs'] = categorical_specs
    with open(fpath, 'w', encoding='UTF-8') as f:
        json.dump(data, f)
    return fpath


def _write_root_manifest(
    fpath: pathlib.Path,
    entries: list[dict[str, typing.Any]],
) -> pathlib.Path:
    '''Write root dataset manifest JSON linking sub-manifest entries.'''
    serializable = [
        {
            'name': e['name'],
            'path': str(e['path']),
            'manifest': str(e['manifest']),
        }
        for e in entries
    ]
    with open(fpath, 'w', encoding='UTF-8') as f:
        json.dump(serializable, f)
    return fpath
