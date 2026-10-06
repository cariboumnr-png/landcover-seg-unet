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
Storage and filesystem diagnostic probes.

Validates output directory access, writability, and path readiness.

Public APIs:
    - `probe_storage`: Validate pipeline output directory access.
'''

# standard imports
import os
# local imports
import landseg.artifacts as artifacts
import landseg.artifacts.paths.base as paths_base
import landseg.configs as configs
import landseg.execution.pipelines.base as base
import landseg.execution.preflight.schema as schema


# ----- public functions
def probe_storage(
    target_or_pipeline: base.BasePipeline | str,
    root_config: configs.RootConfig | None = None,
) -> list[schema.ProbeResult]:
    '''
    Validate output directory writability and storage reachability.

    Args:
        target_or_pipeline:
            pipeline runner instance or target string identifier.
        root_config:
            optional root configuration for storage destinations.

    Returns:
        list[schema.ProbeResult]:
            list of storage diagnostic probe results.
    '''
    probes: list[schema.ProbeResult] = []

    target_dir: str | None = None
    if isinstance(target_or_pipeline, base.BasePipeline):
        target = target_or_pipeline.pipeline_name
        if target_or_pipeline.pipeline_paths is not None:
            p_paths = target_or_pipeline.pipeline_paths
            if isinstance(p_paths, paths_base.PipelineArtifactsPaths):
                target_dir = p_paths.root
    else:
        target = target_or_pipeline
        eff_config = root_config or configs.RootConfig()
        artifact_paths = artifacts.ArtifactPaths.from_config(eff_config)
        target_to_paths = {
            'world-grid': artifact_paths.world_grid.root,
            'data-harmonize': artifact_paths.data_harmonization.root,
            'data-ingest': artifact_paths.data_ingestion.root,
            'data-prepare': artifact_paths.data_preparation.root,
            'model-train': artifact_paths.session.root,
            'model-evaluate': artifact_paths.session.root,
        }
        target_dir = target_to_paths.get(target)

    # check pipeline output directory
    if target_dir is not None:
        check_dir = target_dir
        while check_dir and not os.path.exists(check_dir):
            parent = os.path.dirname(check_dir)
            if parent == check_dir:
                break
            check_dir = parent

        if check_dir and os.path.exists(check_dir):
            is_writable = os.access(check_dir, os.W_OK)
            status = (
                schema.ProbeStatus.PASS
                if is_writable
                else schema.ProbeStatus.FAIL
            )
            msg = (
                f"'{target_dir}' writable"
                if is_writable
                else f"'{target_dir}' not writable"
            )
        else:
            status = schema.ProbeStatus.PASS
            msg = f"'{target_dir}' path resolved"

        probes.append(
            schema.ProbeResult(
                probe_id='output_directory',
                category='storage',
                status=status,
                message=msg,
                details={'target_dir': target_dir},
            )
        )

    # pool directory check for batch ingestion
    if target == 'batch-ingest' and root_config is not None:
        artifact_paths = artifacts.ArtifactPaths.from_config(root_config)
        pool_dir = artifact_paths.data_ingestion.root
        if os.path.exists(pool_dir) and not os.access(pool_dir, os.W_OK):
            probes.append(
                schema.ProbeResult(
                    probe_id='pool_write_access',
                    category='storage',
                    status=schema.ProbeStatus.FAIL,
                    message=f"Pool directory '{pool_dir}' not writable",
                    details={'pool_dir': pool_dir},
                )
            )
        else:
            probes.append(
                schema.ProbeResult(
                    probe_id='pool_write_access',
                    category='storage',
                    status=schema.ProbeStatus.PASS,
                    message=f"Pool destination '{pool_dir}' is writable",
                    details={'pool_dir': pool_dir},
                )
            )

    # database endpoint checks for sweeps and studies
    if target in {'study-sweep', 'study-analysis'} and root_config is not None:
        sweep_cfg = getattr(root_config.command, 'study_sweep', None)
        storage_url = (
            getattr(sweep_cfg, 'storage', 'sqlite:///optuna.db')
            if sweep_cfg
            else 'sqlite:///optuna.db'
        )
        if storage_url.startswith('sqlite:///'):
            db_path = storage_url.removeprefix('sqlite:///')
            db_dir = os.path.dirname(db_path) or '.'
            if os.path.exists(db_dir) and os.access(db_dir, os.W_OK):
                probes.append(
                    schema.ProbeResult(
                        probe_id='rdbms_connection',
                        category='storage',
                        status=schema.ProbeStatus.PASS,
                        message=f'Storage path writable: {storage_url}',
                        details={'storage': storage_url},
                    )
                )
            else:
                probes.append(
                    schema.ProbeResult(
                        probe_id='rdbms_connection',
                        category='storage',
                        status=schema.ProbeStatus.FAIL,
                        message=f'Storage directory not writable: {db_dir}',
                        details={'storage': storage_url},
                    )
                )
        else:
            probes.append(
                schema.ProbeResult(
                    probe_id='rdbms_connection',
                    category='storage',
                    status=schema.ProbeStatus.PASS,
                    message=f'RDBMS endpoint configured: {storage_url}',
                    details={'storage': storage_url},
                )
            )

    return probes
