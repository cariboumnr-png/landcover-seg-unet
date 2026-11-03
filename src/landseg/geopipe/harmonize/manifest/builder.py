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
Fluent programmatic dataset manifest builder.

Provides type-safe construction, validation, and serialization of raw
dataset manifests and sidecar configuration bundles.

Public APIs:
    - `DatasetManifestCompiler`: Compiler for dataset manifests.
'''

# standard imports
import pathlib
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.harmonize.manifest.normalizer as normalizer
import landseg.geopipe.harmonize.manifest.schema as schema
import landseg.utils.validator as val


# ----- public classes
class DatasetManifestError(Exception):
    '''Base class for errors when compiling dataset manifest JSON.'''

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class DatasetManifestBuilder:
    '''
    Fluent programmatic builder for raw dataset manifests.

    Manages feature, domain, and label raster entries, enforces category
    constraints, verifies physical GeoTIFF headers, and compiles both
    canonical JSON bundles and unified single-document specifications.
    '''

    def __init__(
        self,
        entries: list[dict[str, typing.Any]] | None = None
    ) -> None:
        '''
        Initialize manifest builder with an optional base directory.

        Args:
            base_dir:
                Root directory used to resolve relative raster paths.
                Defaults to current working directory.
        '''
        self._entries: list[dict[str, typing.Any]] = entries or []
        self._compiled: list[schema.ManifestEntry] = []

    @property
    def raw_entries(self) -> tuple[dict[str, typing.Any], ...]:
        '''Return uncompiled raster entries registered in builder.'''
        return tuple(self._entries)

    @classmethod
    def from_file(cls, fpath: str | pathlib.Path) -> typing.Self:
        '''
        Construct builder from a manifest file (JSON or YAML).

        Supports either a top-level list of raster entries or a top-level
        mapping containing a "rasters" list. Resolves relative raster paths
        against the manifest file location.

        Args:
            fpath:
                Path to the manifest JSON or YAML file.

        Returns:
            typing.Self:
                Populated DatasetManifestBuilder instance.

        Raises:
            DatasetManifestError:
                If manifest parsing or schema structure validation fails.
        '''
        try:
            _fpath = val.require_file(fpath)
            ctrl = artifacts.Controller[typing.Any](_fpath)
            ctrl.hash(overwrite=False)
            raw = ctrl.fetch()

            if isinstance(raw, dict) and 'rasters' in raw:
                raw_entries = val.require_sequence(raw['rasters'])
                base_dir = _fpath.parent / raw.get('base_dir', '.')
            else:
                raw_entries = val.require_sequence(raw)
                base_dir = _fpath.parent

            _data: list[dict[str, typing.Any]] = []
            for d in raw_entries:
                _d = dict(val.require_mapping(d))
                if 'path' in _d:
                    p = pathlib.Path(_d['path'])
                    if not p.is_absolute():
                        _d['path'] = (base_dir / p).resolve()
                _data.append(_d)

        except Exception as e:
            raise DatasetManifestError(
                f'Error parsing manifest {fpath}: {e}, '
                f'expecting a list of entries or mapping with "rasters"'
            ) from e

        return cls(_data)

    def add_feature(
        self,
        *,
        name: str,
        path: str | pathlib.Path,
        bands: list[str],
        schemes: dict[str, tuple[str]] | None = None,
    ) -> typing.Self:
        '''
        Add a multi-band continuous feature raster layer.

        Args:
            name:
                Unique identifier for the feature raster layer.
            path:
                Path to the GeoTIFF raster file.
            bands:
                Ordered sequence of band names mapped to 1-based indices.
            schemes:
                Optional dictionary of named band subsets.

        Returns:
            typing.Self:
                Builder instance for chained configuration.
        '''
        self._entries.append({
            'name': name,
            'path': path,
            'category': 'features',
            'band_mapping': {idx + 1: b for idx, b in enumerate(bands)},
            'categorical_specs': None,
            'schemes': schemes,
        })
        return self

    def add_domain( # pylint: disable=R0913
        self,
        *,
        name: str,
        path: str | pathlib.Path,
        num_cls: int,
        band_name: str | None = None,
        index_base: int = 1,
        ignore_cls: list[int] | None = None,
    ) -> typing.Self:
        '''
        Add a discrete environmental domain raster layer.

        Args:
            name:
                Unique identifier for the domain raster layer.
            path:
                Path to the single-band domain GeoTIFF file.
            num_cls:
                Number of discrete domain classes.
            band_name:
                Band name identifier (defaults to layer name).
            index_base:
                Base class index (typically 0 or 1, defaults to 1).
            ignore_cls:
                Sequence of class values to mask out (defaults to [255]).

        Returns:
            typing.Self:
                Builder instance for chained configuration.
        '''
        self._entries.append({
            'name': name,
            'path': path,
            'category': 'domains',
            'band_mapping': {1: band_name or name}, # assuming 1-band here
            'categorical_specs': {
                'index_base': index_base,
                'num_cls': num_cls,
                'ignore_cls': ignore_cls or [255],
            },
            'schemes': None,
        })
        return self

    def add_label( # pylint: disable=R0913
        self,
        *,
        name: str,
        path: str | pathlib.Path,
        num_cls: int,
        band_name: str | None = None,
        index_base: int = 1,
        ignore_cls: list[int] | None = None,
        classes: dict[int | str, str] | None = None,
        color_map: dict[int | str, tuple[int, int, int]] | None = None,
        taxonomy_profile: str | None = None,
        schemes: dict[str, typing.Any] | None = None,
    ) -> typing.Self:
        '''
        Add a discrete categorical target label raster layer.

        Args:
            name:
                Unique identifier for the target label layer.
            path:
                Path to the single-band label GeoTIFF file.
            num_cls:
                Number of target label classes.
            band_name:
                Band name identifier (defaults to layer name).
            index_base:
                Base class index (typically 0 or 1, defaults to 1).
            ignore_cls:
                Sequence of class values to mask out (defaults to [255]).
            classes:
                Optional mapping of class indices to human-readable names.
            color_map:
                Optional mapping of class indices to RGB 3-tuples.
            taxonomy_profile:
                Optional named taxonomy profile lookup key.
            schemes:
                Optional reclassification schemes (canonical or shorthand).
            auto_verify_raster:
                If True, verifies that the physical raster exists on disk
                and that it is a single-band raster.

        Returns:
            typing.Self:
                Builder instance for chained configuration.
        '''
        self._entries.append({
            'name': name,
            'path': path,
            'category': 'labels',
            'band_mapping': {1: band_name or name}, # assuming 1-band here
            'categorical_specs': {
                'index_base': index_base,
                'num_cls': num_cls,
                'ignore_cls': ignore_cls or [],
                'class_name': classes,
                'color_map': color_map,
                'taxonomy': {
                    'profile': taxonomy_profile
                } if taxonomy_profile else None
            },
            'schemes': schemes,
        })
        return self

    def compile(self) -> tuple[schema.ManifestEntry, ...]:
        '''
        Validate all configured raster entries against normalizer schemas.

        Returns:
            tuple[normalizer.ManifestEntry, ...]:
                Tuple of canonical, validated manifest entries.

        Raises:
            compiler.DatasetManifestError:
                If entries are empty, duplicate, or fail schema checks.
        '''
        if not self._entries:
            raise DatasetManifestError('Manifest has no raster entries')

        compiled: list[schema.ManifestEntry] = []
        names: set[str] = set()
        fingerprints: set[str] = set()

        for idx, entry in enumerate(self._entries):
            name = entry.get('name', '')
            if name in names:
                raise DatasetManifestError(f'Duplicated raster name: {name}')
            names.add(name)

            try:
                norm = normalizer.normalize_manifest_entry(entry)
                compiled.append(norm)
            except Exception as e:
                raise DatasetManifestError(
                    f'Validation failed for entry {idx} ({name}): {e}'
                ) from e

            fingerprint = artifacts.compute_fingerprint(norm)
            if fingerprint in fingerprints:
                raise DatasetManifestError(f'Duplicated data manifest: {name}')
            fingerprints.add(fingerprint)

        self._compiled = compiled
        return tuple(self._compiled)

    def materialize(
        self,
        save_dir: str | pathlib.Path,
        name: str = 'dataset_manifest',
    ) -> None:
        '''Materialize into a JSON, optional with a readable YAML.'''
        data = self._compiled or self.compile()
        json_path = pathlib.Path(save_dir) / f'{name}.json'
        artifacts.Controller(json_path).persist(data)
        yaml_path = pathlib.Path(save_dir) / f'{name}.yaml'
        artifacts.Controller(yaml_path).persist(data)
