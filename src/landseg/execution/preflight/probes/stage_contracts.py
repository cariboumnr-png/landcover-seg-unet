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
Stage contract diagnostic probes backed by authoritative domain
handlers.

Invokes domain validation routines directly from pipeline stages to
verify contract compliance before execution.

Public APIs:
    - `grid_inputs`: Validate world grid definition inputs.
    - `grid_reference`: Validate world grid reference artifact.
    - `manifest_contract`: Validate dataset manifest contract.
    - `pool_grid_compatibility`: Validate pool grid compatibility.
    - `preparation_contract`: Validate data preparation contract.
'''

# standard imports
import pathlib
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.preflight.schema as schema
import landseg.geopipe.core as geo_core
import landseg.geopipe.grid as grid_mod
import landseg.geopipe.harmonize as harmonize_mod
import landseg.geopipe.ingest as ingest_mod
import landseg.geopipe.prepare as prepare_mod


# ----- public functions
def grid_inputs(
    config: configs.RootConfig,
    pid: str | None = None,
) -> schema.ProbeResult:
    '''
    Validate world grid definition inputs through domain handler.

    Args:
        config:
            Root configuration containing world grid options.
        pid:
            Optional probe identifier override.

    Returns:
        schema.ProbeResult:
            Evaluation result with status PASS or FAIL.
    '''
    grid_cfg = config.data.world_grid
    try:
        grid_specs = grid_mod.validators.grid_reference(grid_cfg)
        return schema.ProbeResult(
            pid=pid or 'grid_inputs',
            category='Contract',
            status=schema.ProbeStatus.PASS,
            message='World grid inputs verified',
            details={**grid_specs.as_dict()},
        )
    except Exception as err:  # pylint: disable=broad-exception-caught
        return schema.ProbeResult(
            pid=pid or 'grid_inputs',
            category='Contract',
            status=schema.ProbeStatus.FAIL,
            message=f'World grid inputs invalid: {err}',
            details={'error': str(err), 'mode': getattr(grid_cfg, 'mode', None)},
        )


def grid_reference(
    report_or_config: configs.RootConfig | pathlib.Path | str,
    pid: str | None = None,
) -> schema.ProbeResult:
    '''
    Validate world grid reference artifact through domain handler.

    Args:
        report_or_config:
            RootConfig instance or direct path to world grid report.
        pid:
            Optional probe identifier override.

    Returns:
        schema.ProbeResult:
            Evaluation result with status PASS or FAIL.
    '''
    if isinstance(report_or_config, configs.RootConfig):
        paths = artifacts.ArtifactPaths.from_config(report_or_config)
        report_fp = paths.world_grid.report
    else:
        report_fp = pathlib.Path(report_or_config)

    try:
        grid, gid, grid_fp = harmonize_mod.validators.grid_reference(report_fp)
        return schema.ProbeResult(
            pid=pid or 'grid_reference',
            category='Contract',
            status=schema.ProbeStatus.PASS,
            message=(
                f'World grid reference verified (grid_id: {gid}, '
                f'tiles: {len(grid)})'
            ),
            details={
                'grid_id': gid,
                'grid_fpath': str(grid_fp),
                'report_fpath': str(report_fp),
            },
        )
    except Exception as err:  # pylint: disable=broad-exception-caught
        return schema.ProbeResult(
            pid=pid or 'grid_reference',
            category='Contract',
            status=schema.ProbeStatus.FAIL,
            message=f'World grid reference validation failed: {err}',
            details={'report_fpath': str(report_fp), 'error': str(err)},
        )


def manifest_contract(
    config: configs.RootConfig,
    pid: str | None = None,
) -> schema.ProbeResult:
    '''
    Validate dataset manifest contract through domain handler.

    Args:
        config:
            Root configuration containing harmonization options.
        pid:
            Optional probe identifier override.

    Returns:
        schema.ProbeResult:
            Evaluation result with status PASS or FAIL.
    '''
    manifest_fp = config.data.harmonization.dataset_manifest
    if not manifest_fp:
        return schema.ProbeResult(
            pid=pid or 'manifest_contract',
            category='Contract',
            status=schema.ProbeStatus.FAIL,
            message='Dataset manifest path is not configured',
        )

    try:
        entries = harmonize_mod.validators.manifest_contract(
            pathlib.Path(manifest_fp)
        )
        return schema.ProbeResult(
            pid=pid or 'manifest_contract',
            category='Contract',
            status=schema.ProbeStatus.PASS,
            message=(
                f'Dataset manifest contract verified '
                f'({len(entries)} valid entries)'
            ),
            details={'dataset_manifest': str(manifest_fp)},
        )
    except Exception as err:  # pylint: disable=broad-exception-caught
        return schema.ProbeResult(
            pid=pid or 'manifest_contract',
            category='Contract',
            status=schema.ProbeStatus.FAIL,
            message=f'Dataset manifest contract invalid: {err}',
            details={'error': str(err), 'dataset_manifest': str(manifest_fp)},
        )


def pool_grid_compatibility(
    paths_or_config: (
        configs.RootConfig | artifacts.ArtifactPaths | pathlib.Path
    ),
    grid: geo_core.GridLayout | None = None,
    pid: str | None = None,
) -> schema.ProbeResult:
    '''
    Validate pool grid compatibility through domain handler.

    Args:
        paths_or_config:
            RootConfig, ArtifactPaths, or direct path to schema JSON.
        grid:
            Optional GridLayout instance if `paths_or_config` is a path.
        pid:
            Optional probe identifier override.

    Returns:
        schema.ProbeResult:
            Evaluation result with status PASS or FAIL.
    '''
    schema_fp: pathlib.Path
    target_grid: geo_core.GridLayout

    if grid is not None and isinstance(paths_or_config, (pathlib.Path, str)):
        schema_fp = pathlib.Path(paths_or_config)
        target_grid = grid
    else:
        if isinstance(paths_or_config, configs.RootConfig):
            paths = artifacts.ArtifactPaths.from_config(paths_or_config)
        else:
            assert isinstance(paths_or_config, artifacts.ArtifactPaths)
            paths = paths_or_config

        schema_fp = paths.data_ingestion.data_blocks.schema
        report_fp = paths.world_grid.report
        if not report_fp.exists():
            return schema.ProbeResult(
                pid=pid or 'pool_grid_compatibility',
                category='Contract',
                status=schema.ProbeStatus.FAIL,
                message=f'World grid report not found at {report_fp}',
                details={'report_fpath': str(report_fp)},
            )
        try:
            target_grid = harmonize_mod.validators.grid_reference(
                report_fp
            )[0]
        except Exception as err:  # pylint: disable=broad-exception-caught
            return schema.ProbeResult(
                pid=pid or 'pool_grid_compatibility',
                category='Contract',
                status=schema.ProbeStatus.FAIL,
                message=f'World grid reference invalid: {err}',
                details={'report_fpath': str(report_fp), 'error': str(err)},
            )

    try:
        ingest_mod.validators.pool_compatibility(
            schema_fp, target_grid
        )
        if not schema_fp.exists():
            msg = 'Pool schema not yet initialized (baseline established)'
        else:
            msg = (
                'Pool grid compatibility verified '
                f'(grid_id: {target_grid.gid})'
            )
        return schema.ProbeResult(
            pid=pid or 'pool_grid_compatibility',
            category='Contract',
            status=schema.ProbeStatus.PASS,
            message=msg,
            details={
                'grid_id': target_grid.gid,
                'schema_fpath': str(schema_fp),
                'schema_exists': schema_fp.exists(),
            },
        )
    except Exception as err:  # pylint: disable=broad-exception-caught
        return schema.ProbeResult(
            pid=pid or 'pool_grid_compatibility',
            category='Contract',
            status=schema.ProbeStatus.FAIL,
            message=f'Pool grid compatibility invalid: {err}',
            details={'schema_fpath': str(schema_fp), 'error': str(err)},
        )


def preparation_contract(
    paths_or_config: (
        configs.RootConfig | artifacts.ArtifactPaths | pathlib.Path
    ),
    config: configs.RootConfig | None = None,
    pid: str | None = None,
) -> schema.ProbeResult:
    '''
    Validate data preparation contract through domain handlers.

    Verifies upstream ingestion schema and catalog, and validates
    requested feature channels and target head configurations against
    the dataset schema.

    Args:
        paths_or_config:
            RootConfig, ArtifactPaths, or direct path to schema JSON.
        config:
            Optional RootConfig if `paths_or_config` is ArtifactPaths.
        pid:
            Optional probe identifier override.

    Returns:
        schema.ProbeResult:
            Evaluation result with status PASS or FAIL.
    '''
    schema_fp: pathlib.Path
    root_cfg: configs.RootConfig | None

    if isinstance(paths_or_config, (pathlib.Path, str)):
        schema_fp = pathlib.Path(paths_or_config)
        root_cfg = config
    elif isinstance(paths_or_config, configs.RootConfig):
        root_cfg = paths_or_config
        paths = artifacts.ArtifactPaths.from_config(root_cfg)
        schema_fp = paths.data_ingestion.data_blocks.schema
    else:
        assert isinstance(paths_or_config, artifacts.ArtifactPaths)
        paths = paths_or_config
        root_cfg = config
        schema_fp = paths.data_ingestion.data_blocks.schema

    if not schema_fp.exists():
        return schema.ProbeResult(
            pid=pid or 'preparation_contract',
            category='Contract',
            status=schema.ProbeStatus.FAIL,
            message=f'Upstream ingestion schema not found at {schema_fp}',
            details={'schema_fpath': str(schema_fp)},
        )

    try:
        Ctrl = artifacts.Controller[geo_core.DataSchema]
        data_schema = Ctrl(
            schema_fp, schema_id=geo_core.data_schema.SCHEMA_ID
        ).fetch()

        if root_cfg is not None:
            prep_cfg = root_cfg.data.preparation
            features = prepare_mod.validators.feature_channels(
                data_schema, prep_cfg.datasetview.features
            )
            targets = prepare_mod.validators.target_heads(
                data_schema, prep_cfg.datasetview.targets
            )
            focal_head = prepare_mod.validators.focal_head(
                targets, prep_cfg.datasetview.focal_target
            )
            for aoi_name, aoi_val in [
                ('train_aoi', prep_cfg.partition.train_aoi),
                ('val_aoi', prep_cfg.partition.val_aoi),
                ('test_aoi', prep_cfg.partition.test_aoi),
            ]:
                if aoi_val and not pathlib.Path(aoi_val).exists():
                    raise FileNotFoundError(
                        f'Configured {aoi_name} not found at {aoi_val}'
                    )

            return schema.ProbeResult(
                pid=pid or 'preparation_contract',
                category='Contract',
                status=schema.ProbeStatus.PASS,
                message=(
                    f'Preparation contract verified (focal_head: {focal_head}, '
                    f'features: {len(features.names)}, '
                    f'target_heads: {len(targets.head_names)})'
                ),
                details={
                    'focal_head': focal_head,
                    'features': list(features.names),
                    'target_heads': list(targets.head_names),
                    'schema_fpath': str(schema_fp),
                },
            )

        return schema.ProbeResult(
            pid=pid or 'preparation_contract',
            category='Contract',
            status=schema.ProbeStatus.PASS,
            message='Upstream ingestion schema verified',
            details={'schema_fpath': str(schema_fp)},
        )
    except Exception as err:  # pylint: disable=broad-exception-caught
        return schema.ProbeResult(
            pid=pid or 'preparation_contract',
            category='Contract',
            status=schema.ProbeStatus.FAIL,
            message=f'Preparation contract invalid: {err}',
            details={'schema_fpath': str(schema_fp), 'error': str(err)},
        )
