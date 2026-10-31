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
Overfit test pipeline.

Constructs a single valid block, builds minimal data specifications,
and trains until near-perfect IoU to validate the end-to-end stack.
'''

# standard imports
import pathlib
import random
import typing
# third-party imports
import numpy
import torch
# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.core as core
import landseg.geopipe.core as geo_core
import landseg.geopipe.ingest as ingest
import landseg.geopipe.ingest.blocks.assembler as assembler
import landseg.geopipe.ingest.blocks.mapper as mapper_mod
import landseg.knowledge as knowledge
import landseg.models as models
import landseg.session as session


# ----- overfit test pipeline
def execute_diagnose_overfit(config: configs.RootConfig) -> None:
    '''
    Run an overfit test on a single block.

    Creates or loads a block, builds a small `DataSpecs`, instantiates
    the model and a trainer with minimal logging, and trains until an
    IoU threshold or the epoch limit is reached.

    Args:
        config: RootConfig with model/trainer settings.
    '''
    root = pathlib.Path(config.execution.exp_root) / 'results' / 'overfit_test'
    log_file = root / 'log' / 'overfit_summary.json'
    logger = session.SessionLogger('overfit', log_file=log_file)
    logger.init_summary(run_id='overfit_test', command='diagnose_overfit')

    try:
        logger.log_sep()
        logger.set_inputs({
            'exp_root': config.execution.exp_root,
            'patch_size': config.session.dataloader.patch_size,
            'lr': config.session.engine_optim.lr,
            'max_epoch': c.OVERFIT_MAX_EPOCH,
        })

        dataspecs = _prepare_dataspecs(root, config, logger)

        model = models.build_multihead_unet(
            patch_size=config.session.dataloader.patch_size,
            dataspecs=dataspecs,
            unet_backbone_config=config.models.unet_backbone_config,
            conditioning_config=config.models.conditioning_config,
            enable_clamp=config.models.numeric_safety.enable_clamp,
            clamp_range=config.models.numeric_safety.clamp_range,
        )

        runner = session.build_session_runner(
            dataspecs=dataspecs,
            model=model,
            config=config.session,
            context=session.SessionBuildContext(
                device=c.DEVICE,
                session_paths=artifacts.ArtifactPaths().session, # no ops
                eval_dataset='val',
                logger=logger
            ),
            session_type='overfit'
        )
        monitor_head = config.session.orchestration.monitor.track_heads
        active_heads = (
            list(monitor_head.keys())
            if monitor_head
            else list(dataspecs.heads.class_counts.keys())
        )
        runner.set_head_state(active_heads=active_heads)

        results = _run_overfit_loop(runner, config, logger)
        status = 'SUCCESS' if results['overfit_reached'] else 'FAILED'

        logger.set_results(results)
        logger.set_status(status)

    except Exception as e:
        logger.set_status('FAILED')
        logger.error(f'Overfit pipeline failed: {e}', exc_info=True)
        raise e

    finally:
        logger.log_sep()
        logger.close()


# ----- dataspecs builder
def _prepare_dataspecs(
    save_dpath: pathlib.Path,
    config: configs.RootConfig,
    logger: session.SessionLogger,
) -> core.DataSpecs:
    '''Build or select a single test block and construct `DataSpecs`.'''
    block_fpath: pathlib.Path | None = None
    if save_dpath.exists():
        for p in save_dpath.iterdir():
            if p.name.endswith('.npz'):
                block_fpath = p
                logger.info(f'Using existing block: {block_fpath}')
                break

    if not block_fpath:
        block_fpath = _create_block(save_dpath, config, logger)

    block = geo_core.DataBlock.load(block_fpath)
    counts = block.manifest['label_count']
    cc = {k: [1] * len(counts[k]) for k in counts if k != 'original'}

    tax: dict[str, dict[str, typing.Any]] = {
        k: dict(v)
        for k, v in block.manifest.get('label_taxonomy', {}).items()
    }
    cls_names = block.manifest.get('label_cls_names', {})
    sim_matrices: dict[str, torch.Tensor] = {}
    kdir = config.execution.knowledge_dir
    kroot = pathlib.Path(kdir) if kdir else None
    if kroot is not None and kroot.exists():
        for hname, tax_dict in tax.items():
            if isinstance(tax_dict, dict) and 'profile' in tax_dict:
                profile = tax_dict['profile']
                cls_list = cls_names.get(hname)
                num_cls = len(counts[hname])
                if cls_list and len(cls_list) == num_cls:
                    mapping = {str(i): c for i, c in enumerate(cls_list, 1)}
                    try:
                        resolved = knowledge.validate_specs(
                            profile,
                            mapping,
                            num_cls,
                            knowledge_root=kroot,
                        )
                        tax[hname] = dict(resolved)
                    except (ValueError, FileNotFoundError):
                        pass
                try:
                    sim_matrices[hname] = knowledge.resolve_similarity_matrix(
                        profile,
                        knowledge_root=kroot,
                    )
                except (FileNotFoundError, artifacts.ArtifactError):
                    pass

    return core.DataSpecs(
        name='overfit_single_block',
        mode='default',
        meta=core.Meta(
            blk_bytes=0,
            test_blks_grid=(0, 0),
            label_color_map=None,
            image_specs=core.Meta.Image(
                num_channels=block.data.image.shape[0],
                height_width=block.data.image.shape[1],  # assume H == W
                array_key='image',
                band_map=block.manifest['image_band_map'],
            ),
            label_specs=core.Meta.Label(
                array_key='label',
                ignore_index=block.manifest['label_ignore_index'],
            ),
        ),
        heads=core.Heads(
            class_counts=cc,  # neutral
            logits_adjust={k: [1.0] * len(v) for k, v in cc.items()},
            head_parent={k: None for k in block.manifest['label_band_map']},
            head_parent_cls={k: None for k in block.manifest['label_band_map']},
            taxonomy=tax,
            similarity_matrices=sim_matrices,
        ),
        splits=core.Splits(
            train={block.manifest['block_name']: block_fpath.resolve()},
            val={block.manifest['block_name']: block_fpath.resolve()},
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


# ----- block construction helper
def _create_block(
    save_dpath: pathlib.Path,
    config: configs.RootConfig,
    logger: session.SessionLogger,
) -> pathlib.Path:
    '''Build one valid block for the overfit test.'''
    artifact_paths = artifacts.ArtifactPaths.from_config(config)
    hm_record = ingest.resolve_pending_ingestion_batches(
        artifact_paths.data_harmonization.runs_manifest,
        artifact_paths.data_ingestion.runs_manifest,
        target='latest',
    )[0]
    context = ingest.build_ingestion_context(
        artifact_paths.data_harmonization,
        hm_record,
    )
    if not context.has_data:
        raise ValueError('Harmonized feature/label rasters not found in report')

    # map raster windows onto world grid
    logger.info('Mapping image unto the world grid')
    datablocks_cfg = config.data.ingestion.datablocks
    assert context.features
    assert context.labels
    mapped = mapper_mod.map_rasters_to_grid(
        context.grid,
        context.features,
        context.labels,
        artifact_paths.data_ingestion.windows,
        policy=artifacts.LifecyclePolicy.REBUILD
    )

    # retrieve band map and label specs from VRT
    logger.info('Building a single data block')
    raster_ctx = assembler.RasterReadContext.from_rasters(
        context.features, context.labels
    )
    if raster_ctx.label_specs is None:
        raise ValueError('Label specifications not found in label raster')

    # resolve target head for filtering
    target_head = _resolve_target_head(config, raster_ctx.label_specs)

    # build single valid test block matching criteria
    logger.debug('Try: valid_px_per=0.95; need_all_class=True')
    block_fpath = _build_test_block(
        save_dpath=save_dpath,
        mapped=mapped,
        raster_ctx=raster_ctx,
        dem_pad_px=datablocks_cfg.image_dem_pad,
        target_head=target_head,
        valid_px_per=0.95,
        need_all_classes=True,
    )
    # second try
    if not block_fpath:
        logger.debug('Try: valid_px_per=0.95; need_all_class=False')
        block_fpath = _build_test_block(
            save_dpath=save_dpath,
            mapped=mapped,
            raster_ctx=raster_ctx,
            dem_pad_px=datablocks_cfg.image_dem_pad,
            target_head=target_head,
            valid_px_per=0.95,
            need_all_classes=False,
        )
    # fail
    if not block_fpath:
        raise ValueError('No valid block for testing is found')

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
    '''Build, normalize, and persist a single valid block for testing.'''
    coords = list(mapped.image.keys())
    random.Random(42).shuffle(coords)

    selected_name = None
    selected_candidate = None

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
            candidate = assembler.build_data_block(
                name,
                raster_data,
            )
            manifest = candidate.manifest

            # check valid pixel ratios based on image band
            valid_ratio = manifest['valid_ratios'].get('image', 0.0)

            # check if all classes are present
            has_all_classes = True
            if target_head in manifest['label_count']:
                has_all_classes = all(manifest['label_count'][target_head])
            else:
                has_all_classes = False

            if (
                valid_ratio >= valid_px_per and
                (has_all_classes or not need_all_classes)
            ):
                selected_name = name
                selected_candidate = candidate
                break

        except ValueError:
            continue

    if selected_candidate is None:
        return None

    candidate = selected_candidate
    name = selected_name

    # in-place image normalization for debugging block
    mean = numpy.mean(candidate.data.image)
    std = numpy.std(candidate.data.image)
    candidate.data.image = (candidate.data.image - mean) / (std or 1.0)

    dpath = pathlib.Path(save_dpath)
    dpath.mkdir(parents=True, exist_ok=True)
    fpath = dpath / f'test_{name}.npz'
    candidate.save(fpath)
    return fpath


# ----- target head resolution helper
def _resolve_target_head(
    config: configs.RootConfig,
    label_specs: dict[str, geo_core.CategoricalSpec],
) -> str:
    '''Resolve the target head for test block filtering.'''
    if not label_specs:
        raise ValueError('No label specifications found')

    first_head = list(label_specs.keys())[0]
    active_heads = config.session.orchestration.single_phase.active_heads

    return active_heads[0] if active_heads else first_head


# ----- overfit epoch training loop helper
def _run_overfit_loop(
    runner: session.EpochRunner,
    config: configs.RootConfig,
    logger: session.SessionLogger,
) -> dict[str, typing.Any]:
    '''Execute epoch training loop until threshold or max epochs.'''
    max_epoch = c.OVERFIT_MAX_EPOCH
    lr = config.session.engine_optim.lr
    logger.info('Starting overfit test')
    logger.info(f'Maximum epoch: {max_epoch}')
    logger.info(f'Learning rate: {lr}')

    loss, iou = 0.0, 0.0
    for ep in range(1, max_epoch + 1):
        results = runner.run_epoch(ep)
        assert results.training and results.validation  # typing
        results.track(
            config.session.orchestration.monitor.metric_name,
            config.session.orchestration.monitor.track_heads,
        )
        loss = results.training.total_objective
        iou = results.target_metrics
        logger.info(f'Epoch: {ep:04d} | Loss: {loss:.4f} | IoU: {iou:.4f}')
        if iou >= 0.99:
            logger.info('Overfit reached - test complete')
            return {
                'final_epoch': ep,
                'final_loss': loss,
                'final_iou': iou,
                'overfit_reached': True,
            }

    logger.warning(f'IoU did not reach 99% after {max_epoch} epochs.')
    return {
        'final_epoch': max_epoch,
        'final_loss': loss,
        'final_iou': iou,
        'overfit_reached': False,
    }
