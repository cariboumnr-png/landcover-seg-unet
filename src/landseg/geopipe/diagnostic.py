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
Diagnostic single-block `DataSpecs` builder for overfit testing.

Selects or assembles a representative normalized `DataBlock` directly
from the latest harmonized rasters and adapts it into a runtime
`core.DataSpecs` contract without exposing `geopipe` or `knowledge`
internals to workflows.

Public APIs:
    - `build_overfit_dataspec`: Build single-block overfit `DataSpecs`.
'''

# standard imports
import pathlib
import random
import typing
# third-party imports
import numpy
# local imports
import landseg.artifacts as artifacts
import landseg.core as core
import landseg.geopipe.core as geo_core
import landseg.geopipe.ingest as ingest
import landseg.geopipe.ingest.blocks.assembler as assembler
import landseg.geopipe.ingest.blocks.mapper as mapper_mod
import landseg.knowledge as knowledge
import landseg.runtime as runtime


# ----- public functions
def build_overfit_dataspec(
    save_dpath: pathlib.Path,
    harmonization_paths: artifacts.HarmonizationPaths,
    ingestion_paths: artifacts.IngestionPaths,
    *,
    dem_pad_px: int = 8,
    active_heads: typing.Sequence[str] | None = None,
    knowledge_root: pathlib.Path | str | None = None,
    logger: runtime.Logger | None = None,
) -> core.DataSpecs:
    '''
    Load or construct one valid block and assemble `DataSpecs`.

    Args:
        save_dpath:
            Directory to search for or persist the diagnostic block.
        harmonization_paths:
            Path manager for upstream harmonization artifacts.
        ingestion_paths:
            Path manager for ingestion artifacts and window caches.
        dem_pad_px:
            Context padding in pixels around DEM windows.
        active_heads:
            Optional ordered sequence of active target head names.
        knowledge_root:
            Optional ecological knowledge root for similarity matrices.
        logger:
            Optional logger for diagnostic creation status messages.

    Returns:
        core.DataSpecs:
            Runtime dataset specs for single-block overfit testing.
    '''
    block_fpath = _find_existing_block(save_dpath, logger)
    if block_fpath is None:
        block_fpath = _create_diagnostic_block(
            save_dpath=save_dpath,
            harmonization_paths=harmonization_paths,
            ingestion_paths=ingestion_paths,
            dem_pad_px=dem_pad_px,
            active_heads=active_heads,
            logger=logger,
        )

    return _assemble_overfit_dataspec(block_fpath, knowledge_root)


# ----- private helpers
def _assemble_overfit_dataspec(
    block_fpath: pathlib.Path,
    knowledge_root: pathlib.Path | str | None,
) -> core.DataSpecs:
    '''Load cached diagnostic block and build `core.DataSpecs`.'''
    block = geo_core.DataBlock.load(block_fpath)
    manifest = block.manifest
    counts = manifest['label_count']
    neutral_counts = {
        k: [1] * len(counts[k]) for k in counts if k != 'original'
    }
    taxonomy, sim_matrices = knowledge.resolve_head_knowledge(
        counts,
        manifest.get('label_taxonomy', {}),
        manifest.get('label_cls_names', {}),
        knowledge_root,
    )

    resolved_fp = block_fpath.resolve()
    block_name = manifest['block_name']
    label_heads = manifest['label_band_map']
    return core.DataSpecs(
        name='overfit_single_block',
        mode='default',
        meta=core.Meta(
            blk_bytes=0,
            test_blks_grid=(0, 0),
            label_color_map=None,
            image_specs=core.Meta.Image(
                height_width=block.data.image.shape[1],
                array_key='image',
                band_map=manifest['image_band_map'],
            ),
            label_specs=core.Meta.Label(
                array_key='label',
                ignore_index=manifest['label_ignore_index'],
            ),
        ),
        heads=core.Heads(
            class_counts=neutral_counts,
            logits_adjust={
                k: [1.0] * len(v) for k, v in neutral_counts.items()
            },
            head_parent={k: None for k in label_heads},
            head_parent_cls={k: None for k in label_heads},
            taxonomy=taxonomy,
            similarity_matrices=sim_matrices,
        ),
        splits=core.Splits(
            train={block_name: resolved_fp},
            val={block_name: resolved_fp},
            test={},
        ),
        domains=core.Domains(
            train={'ids_domain': None, 'vec_domain': None},
            val={'ids_domain': None, 'vec_domain': None},
            test={'ids_domain': None, 'vec_domain': None},
            ids_num=0,
            vec_dim=0,
        ),
    )
def _find_existing_block(
    save_dpath: pathlib.Path,
    logger: runtime.Logger | None,
) -> pathlib.Path | None:
    '''Return cached diagnostic block in `save_dpath` if present.'''
    if not save_dpath.exists():
        return None
    for entry in sorted(save_dpath.iterdir()):
        if entry.name.endswith('.npz'):
            if logger is not None:
                logger.info(f'Using existing block: {entry}')
            return entry
    return None


def _create_diagnostic_block(
    save_dpath: pathlib.Path,
    harmonization_paths: artifacts.HarmonizationPaths,
    ingestion_paths: artifacts.IngestionPaths,
    *,
    dem_pad_px: int,
    active_heads: typing.Sequence[str] | None,
    logger: runtime.Logger | None,
) -> pathlib.Path:
    '''Build one normalized block from the latest harmonization run.'''
    pending = ingest.resolve_pending_ingestion_batches(
        harmonization_paths.runs_manifest,
        ingestion_paths.runs_manifest,
        target='latest',
    )
    if not pending:
        raise ValueError(
            'No pending harmonization runs found for overfit block creation'
        )
    context = ingest.build_ingestion_context(harmonization_paths, pending[0])
    if not context.has_data or not context.features or not context.labels:
        raise ValueError('Harmonized feature/label rasters not found in report')

    if logger is not None:
        logger.info('Mapping image unto the world grid')
    mapped = mapper_mod.map_rasters_to_grid(
        context.grid,
        context.features,
        context.labels,
        ingestion_paths.windows,
        policy=artifacts.LifecyclePolicy.REBUILD,
    )

    if logger is not None:
        logger.info('Building a single data block')
    raster_ctx = assembler.RasterReadContext.from_rasters(
        context.features, context.labels
    )
    if raster_ctx.label_specs is None:
        raise ValueError('Label specifications not found in label raster')

    target_head = _resolve_target_head(raster_ctx.label_specs, active_heads)
    block_fpath: pathlib.Path | None = None
    for need_all in (True, False):
        if logger is not None:
            logger.debug(
                f'Try: valid_px_per=0.95; need_all_class={need_all}'
            )
        block_fpath = _build_test_block(
            save_dpath=save_dpath,
            mapped=mapped,
            raster_ctx=raster_ctx,
            dem_pad_px=dem_pad_px,
            target_head=target_head,
            valid_px_per=0.95,
            need_all_classes=need_all,
        )
        if block_fpath is not None:
            break

    if block_fpath is None:
        raise ValueError('No valid block for testing is found')

    if logger is not None:
        logger.info(f'Single block successfully created: {block_fpath}')
    return block_fpath


def _build_test_block(
    save_dpath: pathlib.Path,
    mapped: mapper_mod.MappedRasterWindows,
    raster_ctx: assembler.RasterReadContext,
    dem_pad_px: int,
    *,
    target_head: str,
    valid_px_per: float,
    need_all_classes: bool,
) -> pathlib.Path | None:
    '''Build, normalize, and persist one valid block for testing.'''
    coords = list(mapped.image.keys())
    random.Random(42).shuffle(coords)

    selected_name: str | None = None
    selected_candidate: geo_core.DataBlock | None = None

    for coord in coords:
        name = geo_core.xy_name(coord)
        img_win = mapped.image[coord]
        lbl_win = mapped.label[coord] if mapped.label else None

        try:
            raster_data = assembler.read_block_raster_data(
                img_win,
                lbl_win,
                raster_ctx,
                dem_pad=dem_pad_px,
            )
            candidate = assembler.build_data_block(name, raster_data)
        except ValueError:
            continue

        manifest = candidate.manifest
        valid_ratio = manifest['valid_ratios'].get('image', 0.0)
        counts = manifest['label_count']
        has_all_classes = (
            all(counts[target_head]) if target_head in counts else False
        )
        if valid_ratio >= valid_px_per and (
            has_all_classes or not need_all_classes
        ):
            selected_name = name
            selected_candidate = candidate
            break

    if selected_candidate is None or selected_name is None:
        return None

    mean = numpy.mean(selected_candidate.data.image)
    std = numpy.std(selected_candidate.data.image)
    selected_candidate.data.image = (
        selected_candidate.data.image - mean
    ) / (std or 1.0)

    dpath = pathlib.Path(save_dpath)
    dpath.mkdir(parents=True, exist_ok=True)
    fpath = dpath / f'test_{selected_name}.npz'
    selected_candidate.save(fpath)
    return fpath


def _resolve_target_head(
    label_specs: dict[str, geo_core.CategoricalSpec],
    active_heads: typing.Sequence[str] | None,
) -> str:
    '''Resolve the target head for test block filtering.'''
    if not label_specs:
        raise ValueError('No label specifications found')

    first_head = list(label_specs.keys())[0]
    return active_heads[0] if active_heads else first_head
