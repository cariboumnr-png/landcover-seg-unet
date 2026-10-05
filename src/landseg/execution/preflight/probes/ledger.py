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
Ledger and lineage integrity diagnostic probes.

Inspects ETL run manifests (`harmonization_runs.json`, `ingestion_runs.json`)
and discovers pending batches or lineage discrepancies prior to execution.

Public APIs:
    - `probe_ledger`: Inspect ledger health and detect pending runs.
'''

# standard imports
import json
import os
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.preflight.schema as schema
import landseg.geopipe.ingest as ingest


# ----- public functions
def probe_ledger(
    target: str,
    root_config: configs.RootConfig,
) -> list[schema.ProbeResult]:
    '''
    Inspect run ledgers and verify multi-stage execution lineage.

    Args:
        target:
            pipeline or workflow target identifier.
        root_config:
            hydra-composed root configuration.

    Returns:
        list[schema.ProbeResult]:
            list of ledger diagnostic probe records.
    '''
    probes: list[schema.ProbeResult] = []
    artifact_paths = artifacts.ArtifactPaths.from_config(root_config)
    harm_manifest = artifact_paths.data_harmonization.runs_manifest
    ingest_manifest = artifact_paths.data_ingestion.runs_manifest

    # check harmonization ledger
    if target in {'data-harmonize', 'data-ingest', 'batch-ingest', 'model-train'}:
        if os.path.isfile(harm_manifest):
            try:
                with open(harm_manifest, 'r', encoding='utf-8') as f:
                    harm_data = json.load(f)
                run_count = len(harm_data) if isinstance(harm_data, list) else 0
                probes.append(
                    schema.ProbeResult(
                        probe_id='harmonize_ledger',
                        category='ledger',
                        status=schema.ProbeStatus.PASS,
                        message=(
                            f"'{os.path.basename(harm_manifest)}' healthy "
                            f"({run_count} runs)"
                        ),
                        details={'run_count': run_count, 'path': harm_manifest},
                    )
                )
            except Exception as err:  # pylint: disable=broad-exception-caught
                probes.append(
                    schema.ProbeResult(
                        probe_id='harmonize_ledger',
                        category='ledger',
                        status=schema.ProbeStatus.FAIL,
                        message=f'Failed reading {harm_manifest}: {err}',
                        details={'error': str(err)},
                    )
                )
        elif target in {'data-ingest', 'batch-ingest'}:
            probes.append(
                schema.ProbeResult(
                    probe_id='harmonize_ledger',
                    category='ledger',
                    status=schema.ProbeStatus.WARN,
                    message=f'Harmonization ledger not found at {harm_manifest}',
                    details={'path': harm_manifest},
                )
            )

    # pending data warning for model training
    if target == 'model-train':
        try:
            pending = ingest.resolve_pending_ingestion_batches(
                harm_manifest,
                ingest_manifest,
                target=root_config.data.ingestion.harmonization_run,
                rebuild=root_config.data.ingestion.rebuild,
            )
            if pending:
                run_ids = [p['run_id'] for p in pending]
                probes.append(
                    schema.ProbeResult(
                        probe_id='pending_data_warning',
                        category='lineage',
                        status=schema.ProbeStatus.WARN,
                        message=(
                            f'{len(pending)} batches pending in '
                            f'harmonization ledger'
                        ),
                        details={'pending_run_ids': run_ids},
                    )
                )
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    # batch ingestion queue and policy checks
    if target == 'batch-ingest':
        try:
            pending = ingest.resolve_pending_ingestion_batches(
                harm_manifest,
                ingest_manifest,
                target=root_config.data.ingestion.harmonization_run,
                rebuild=root_config.data.ingestion.rebuild,
            )
            run_ids = [p['run_id'] for p in pending]
            probes.append(
                schema.ProbeResult(
                    probe_id='pending_batch_queue',
                    category='lineage',
                    status=schema.ProbeStatus.PASS,
                    message=f'{len(pending)} batches pending in queue',
                    details={'pending_run_ids': run_ids},
                )
            )
        except Exception as err:  # pylint: disable=broad-exception-caught
            probes.append(
                schema.ProbeResult(
                    probe_id='pending_batch_queue',
                    category='lineage',
                    status=schema.ProbeStatus.FAIL,
                    message=f'Failed resolving batch queue: {err}',
                )
            )

        policy = getattr(root_config.data.ingestion, 'collision_policy', 'skip')
        if policy == 'overwrite':
            probes.append(
                schema.ProbeResult(
                    probe_id='collision_policy',
                    category='policy',
                    status=schema.ProbeStatus.WARN,
                    message="Policy 'overwrite' replaces existing blocks",
                    details={'policy': policy},
                )
            )
        else:
            probes.append(
                schema.ProbeResult(
                    probe_id='collision_policy',
                    category='policy',
                    status=schema.ProbeStatus.PASS,
                    message=f"Collision policy '{policy}' configured",
                    details={'policy': policy},
                )
            )

    return probes
