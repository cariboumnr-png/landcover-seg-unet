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
Stage contract diagnostic probes backed by authoritative domain handlers.

Invokes domain validation routines directly from pipeline stages to
verify contract compliance before execution.

Public APIs:
    - `grid_reference`: Validate world grid reference artifact.
    - `manifest_contract`: Validate dataset manifest contract.
'''

# standard imports
import pathlib
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.preflight.schema as schema
import landseg.geopipe.harmonize.handlers as harmonize_handlers


# ----- public functions
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
        grid, gid, grid_fp = harmonize_handlers.validate_grid_reference(report_fp)
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
        entries = harmonize_handlers.validate_manifest_contract(
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
