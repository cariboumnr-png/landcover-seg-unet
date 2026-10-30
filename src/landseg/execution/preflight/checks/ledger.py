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
Canonical pool and prepared-state ledger checks.

Reports the current state of the ingested block pool and prepared
blocks artifacts. These are informational state probes, distinct from
lineage (which asserts upstream readiness).

Public APIs:
    - `pool_state`: Inspect canonical block pool catalog state.
    - `prepared_blocks_state`: Inspect prepared blocks state.
'''

# local imports
import landseg.artifacts as artifacts
import landseg.execution.preflight.context as context
import landseg.execution.preflight.schema as schema


# ----- constants
_CATEGORY = schema.ProbeCategory.LEDGER


# ----- public functions
def ingested_pool_state(
    ctx: context.InspectionContext,
    pid: str = 'ingested_blocks_pool',
) -> schema.ProbeResult:
    '''Inspect canonical block pool catalog state and block count.'''
    catalog_path = ctx.paths.data_ingestion.data_blocks.catalog
    catalog_str = str(catalog_path.resolve())

    if not catalog_path.exists():
        return schema.ProbeResult(
            pid=pid,
            category=_CATEGORY,
            status=schema.ProbeStatus.PASS,
            message='Canonical pool is empty (no blocks ingested yet)',
            details={'path': catalog_str, 'block_count': 0},
        )

    try:
        catalog = artifacts.Controller[dict](catalog_path).fetch()
    except Exception as err:  # pylint: disable=broad-exception-caught
        return schema.ProbeResult(
            pid=pid,
            category=_CATEGORY,
            status=schema.ProbeStatus.FAIL,
            message=f'Failed reading pool catalog: {err}',
            details={'path': catalog_str, 'error': str(err)},
        )

    block_count = len(catalog) if catalog is not None else 0
    return schema.ProbeResult(
        pid=pid,
        category=_CATEGORY,
        status=schema.ProbeStatus.PASS,
        message=f'Ingestion pool contains {block_count} existing blocks',
        details={'path': catalog_str, 'block_count': block_count},
    )


def prepared_blocks_state(
    ctx: context.InspectionContext,
    pid: str = 'prepared_blocks_state',
    *,
    desire_existing: bool = False,
) -> schema.ProbeResult:
    '''Inspect prepared blocks state and split counts.'''
    schema_path = ctx.paths.data_preparation.schema
    try:
        prep_schema = artifacts.Controller[dict](schema_path).fetch()
    except artifacts.ArtifactError:
        prep_schema = None

    match bool(prep_schema), desire_existing:
        case False, False:
            return schema.ProbeResult(
                pid=pid,
                category=_CATEGORY,
                status=schema.ProbeStatus.PASS,
                message='Data preparation not yet run',
            )
        case False, True:
            return schema.ProbeResult(
                pid=pid,
                category=_CATEGORY,
                status=schema.ProbeStatus.WARN,
                message='Data preparation schema not found; blocks not prepared',
            )
        case True, False:
            assert prep_schema is not None
            counts = _split_counts(prep_schema)
            return schema.ProbeResult(
                pid=pid,
                category=_CATEGORY,
                status=schema.ProbeStatus.WARN,
                message=(
                    f'Found existing prepared blocks; '
                    f'train: {counts["train"]} | '
                    f'val: {counts["val"]} | '
                    f'test: {counts["test"]}'
                ),
            )
        case True, True:
            assert prep_schema is not None
            counts = _split_counts(prep_schema)
            return schema.ProbeResult(
                pid=pid,
                category=_CATEGORY,
                status=schema.ProbeStatus.PASS,
                message=(
                    f'Found {counts["train"]} train / {counts["val"]} val / '
                    f'{counts["test"]} test prepared blocks ready'
                ),
            )


# ----- private helpers
def _split_counts(prep_schema: dict) -> dict[str, int]:
    '''Return per-split prepared block counts from a prep schema.'''
    return {
        'train': len(prep_schema.get('train_blocks', {})),
        'val': len(prep_schema.get('val_blocks', {})),
        'test': len(prep_schema.get('test_blocks', {})),
    }
