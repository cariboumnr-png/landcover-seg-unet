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
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        #
#                                   implied.                                  #
#       See the License for the specific language governing permissions       #
#                       and limitations under the License.                    #
# =========================================================================== #

'''Unit tests for prepared-schema ingestion lineage forwarding.'''

# standard imports
import hashlib
import json
import pathlib
import shutil
import typing
# third-party imports
import pytest
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.factory as factory


# ----- `build_dataspec` lineage tests
def test_dataspec_uses_forwarded_ingestion_schema(tmp_path):
    '''
    Given: A prepared schema recording its ingestion schema provenance.
    When: Building `DataSpecs` with only the preparation directory.
    Then: Resolve the ingestion schema and dataset name from provenance.
    '''
    ing = _write_ingestion(tmp_path / 'lakehouse', 'lakehouse')
    prep = tmp_path / 'run_42'
    _write_prepared(prep, ing, relpath='../lakehouse/data_blocks/schema.json')

    specs = factory.build_dataspec(_prep_paths(prep), mode='default')

    assert specs.name == 'lakehouse'
    assert specs.splits.train


def test_dataspec_explicit_ingestion_override(tmp_path):
    '''
    Given: Recorded provenance plus an explicit ingestion override.
    When: Building `DataSpecs` with `ingestion_dpath`.
    Then: The override wins over the forwarded path.
    '''
    ing = _write_ingestion(tmp_path / 'lakehouse', 'lakehouse')
    other = _write_ingestion(tmp_path / 'other', 'override')
    prep = tmp_path / 'run_42'
    _write_prepared(prep, ing)

    specs = factory.build_dataspec(
        _prep_paths(prep),
        mode='default',
        ingestion_dpath=other.parent.parent,
    )

    assert specs.name == 'override'


def test_dataspec_relocated_prepared_bundle(tmp_path):
    '''
    Given: A self-contained bundle whose recorded absolute path is stale.
    When: Building `DataSpecs` from the relocated copy.
    Then: Resolve through the relative provenance and block fallback.
    '''
    ing = _write_ingestion(tmp_path / 'lakehouse', 'lakehouse')
    prep = tmp_path / 'run_42'
    _write_prepared(
        prep,
        ing,
        recorded=str(tmp_path / 'gone' / 'schema.json'),
        relpath='../lakehouse/data_blocks/schema.json',
    )

    bundle = tmp_path / 'bundle'
    moved = bundle / 'prepared_v1'
    shutil.copytree(prep, moved)
    shutil.copytree(ing.parent.parent, bundle / 'lakehouse')
    shutil.rmtree(prep)
    shutil.rmtree(tmp_path / 'lakehouse')

    specs = factory.build_dataspec(_prep_paths(moved), mode='default')

    assert specs.name == 'lakehouse'
    assert specs.splits.train['row_000000_col_000000'].is_file()


def test_dataspec_rejects_checksum_mismatch(tmp_path):
    '''
    Given: Forwarded provenance with a tampered checksum.
    When: Building `DataSpecs` with checksum verification enabled.
    Then: Raise an `ArtifactError` instead of trusting the artifact.
    '''
    ing = _write_ingestion(tmp_path / 'lakehouse', 'lakehouse')
    prep = tmp_path / 'run_42'
    _write_prepared(prep, ing, checksum='0' * 64)

    with pytest.raises(artifacts.ArtifactError, match='checksum mismatch'):
        factory.build_dataspec(
            _prep_paths(prep),
            mode='default',
            verify_ingestion_checksum=True,
        )


def test_dataspec_tolerates_updated_ingestion_schema(tmp_path):
    '''
    Given: A prepared schema whose recorded digest is stale.
    When: Building `DataSpecs` with default settings.
    Then: Resolve the schema, since incremental ingestion updates the
        schema manifest while the forwarded block pool stays valid.
    '''
    ing = _write_ingestion(tmp_path / 'lakehouse', 'lakehouse')
    prep = tmp_path / 'run_42'
    _write_prepared(prep, ing, checksum='0' * 64)

    specs = factory.build_dataspec(_prep_paths(prep), mode='default')

    assert specs.name == 'lakehouse'


def test_dataspec_legacy_schema_requires_fallback(tmp_path):
    '''
    Given: A v0.1.1 prepared schema without provenance.
    When: Building `DataSpecs` with and without a legacy ingestion root.
    Then: Fail informatively first, then resolve via the fallback.
    '''
    ing = _write_ingestion(tmp_path / 'lakehouse', 'lakehouse')
    prep = tmp_path / 'run_42'
    _write_prepared(prep, None, version='prepared_schema/v0.1.1')

    with pytest.raises(FileNotFoundError, match='no ingestion provenance'):
        factory.build_dataspec(_prep_paths(prep), mode='default')

    specs = factory.build_dataspec(
        _prep_paths(prep),
        mode='default',
        legacy_ingestion_dpath=ing.parent.parent,
    )
    assert specs.name == 'lakehouse'


# ----- prepared schema provenance test
def test_prepared_schema_records_ingestion_provenance(tmp_path):
    '''
    Given: A preparation run with a resolved ingestion schema.
    When: Building the prepared schema artifact.
    Then: Persist the ingestion path, relative path, and checksum.
    '''
    from landseg.geopipe.prepare.materialize import schema as prep_schema

    ing = _write_ingestion(tmp_path / 'lakehouse', 'lakehouse')
    prep = tmp_path / 'prepared'
    prep.mkdir()
    paths = _prep_paths(prep)
    _persist(paths.splits_source_blocks, {'train': {}, 'val': {}, 'test': {}})
    _persist(paths.splits_prepared_blocks, {'train': {}, 'val': {}, 'test': {}})
    _persist(paths.label_stats, {'landcover': [1, 1]})
    _persist(paths.image_stats, {})
    logger = _StubLogger()

    prep_schema.build_schema(
        paths,
        _StubContext(),
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
        logger=logger,
        ingestion_schema_fpath=ing,
    )

    written = json.loads(paths.schema.read_text(encoding='utf-8'))
    assert written['artifacts']['ingestion_schema'] == str(ing.resolve())
    assert written['checksums']['ingestion_schema'] == _sha256(ing)
    assert written['artifacts']['ingestion_schema_relpath']
    assert logger.report['ingestion_schema_filepath'] == str(ing.resolve())


# ----- private helpers
def _sha256(path: pathlib.Path) -> str:
    '''Return the SHA-256 digest of a file.'''
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _persist(path: pathlib.Path, payload: typing.Any) -> None:
    '''Persist a JSON artifact through the controller hash registry.'''
    artifacts.Controller[typing.Any](path).persist(payload)


def _prep_paths(prep_root: pathlib.Path) -> artifacts.PreparationPaths:
    '''Return preparation paths for a root.'''
    return artifacts.PreparationPaths(root=prep_root)


def _write_ingestion(root: pathlib.Path, name: str) -> pathlib.Path:
    '''Write a minimal ingestion schema and return its path.'''
    schema_fpath = root / 'data_blocks' / 'schema.json'
    _persist(schema_fpath, _data_schema(name))
    (root / 'domain_knowledge').mkdir(parents=True, exist_ok=True)
    return schema_fpath


def _data_schema(name: str) -> typing.Any:
    '''Build a minimal valid ingestion `DataSchema`.'''
    return {
        'schema_id': geo_core.data_schema.SCHEMA_ID,
        'dataset': {
            'name': name,
            'last_updated': '',
            'dataprep_commit': '',
            'mapped_grids': [],
            'block_identity': 'grid|x',
            'data_source': {
                'image_paths': ['/virtual/image.tif'],
                'label_paths': [],
            },
            'image_schemes': {},
            'label_schemes': {},
        },
        'io_conventions': {
            'block_format': 'npz',
            'shapes': {'image_order': 'C,H,W', 'label_order': 'L,H,W'},
            'dtypes': {'image': 'float32', 'label': 'uint8'},
            'image_band_map': {'red': 0},
            'label_band_map': {'landcover': 0},
            'ignore_index': 255,
        },
        'tensor_shapes': {
            'image': {
                'order': 'C,H,W', 'shape': [1, 8, 8],
                'C': 1, 'H': 8, 'W': 8,
            },
            'label': {
                'order': 'L,H,W', 'shape': [1, 8, 8],
                'L': 1, 'H': 8, 'W': 8,
            },
        },
        'labels': {
            'label_num_cls': {'landcover': 2},
            'label_ignore_cls': {'landcover': [255]},
            'label_class_names': {'landcover': ['a', 'b']},
            'label_class_color_map': None,
            'label_taxonomy': {},
        },
    }


def _write_prepared(
    prep_root: pathlib.Path,
    ingestion_schema: pathlib.Path | None,
    *,
    version: str = 'prepared_schema/v0.1.2',
    recorded: str | None = None,
    relpath: str | None = None,
    checksum: str | None = None,
) -> None:
    '''Write a prepared schema and one dummy block.'''
    blocks = prep_root / 'train_blocks'
    block = blocks / 'row_000000_col_000000.npz'
    block.parent.mkdir(parents=True, exist_ok=True)
    block.write_bytes(b'block')

    artifacts_map: dict[str, str] = {}
    checksums: dict[str, str] = {}
    if ingestion_schema is not None:
        artifacts_map['ingestion_schema'] = (
            recorded or str(ingestion_schema)
        )
        checksums['ingestion_schema'] = checksum or _sha256(ingestion_schema)
    if relpath is not None:
        artifacts_map['ingestion_schema_relpath'] = relpath

    _persist(prep_root / 'schema.json', {
        'schema_id': version,
        'creation_time': '',
        'artifacts': artifacts_map,
        'checksums': checksums,
        'train_blocks': {'row_000000_col_000000': str(block)},
        'val_blocks': {},
        'test_blocks': {},
        'label_stats': {'landcover': [1, 1]},
        'image_stats': {},
        'image_array_key': 'image',
        'label_array_key': 'label',
        'image_band_mapping': {'red': 0},
        'heads': {
            'head_names': ['landcover'],
            'head_parent': {'landcover': None},
            'head_parent_cls': {'landcover': None},
            'num_classes': {'landcover': 2},
            'class_names': {'landcover': ['a', 'b']},
            'ignore_classes': {'landcover': [255]},
        },
    })


class _StubTargets:
    '''Minimal target-heads context for schema generation.'''
    head_names = ['landcover']
    head_parent = {'landcover': None}
    head_parent_cls = {'landcover': None}
    num_classes = {'landcover': 2}
    class_names = {'landcover': ['a', 'b']}
    ignore_classes = {'landcover': [255]}


class _StubFeatures:
    '''Minimal feature selection for schema generation.'''
    image_band_mapping = {'red': 0}


class _StubContext:
    '''Minimal dataset view for schema generation.'''
    targets = _StubTargets()
    features = _StubFeatures()


class _StubLogger:
    '''Capture the schema report without touching disk.'''
    def __init__(self):
        self.report: dict[str, typing.Any] | None = None

    def info(self, *_args, **_kwargs):
        '''Ignore progress logging.'''

    def set_schema(self, report):
        '''Record the schema report.'''
        self.report = report
