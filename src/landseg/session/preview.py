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
Session preview inference utilities.

Executes checkpoint-backed model inference over either a contiguous
test mosaic or a deterministically selected representative fallback
block, returning stitched tensors via `core.InferStepResults`.

Public APIs:
    - `run_preview_inference`: Run inference over test mosaic or block.
    - `select_representative_block`: Select a diverse fallback block.
'''

# standard imports
import dataclasses
import pathlib
# third-party imports
import numpy
# local imports
import landseg.artifacts as artifacts
import landseg.core as core
import landseg.runtime as runtime
import landseg.session.data as data
import landseg.session.engine as engine_mod
import landseg.session.factory as factory
import landseg.session.instrumentation as instrumentation
import landseg.session.logger as session_logger


# ----- public functions
def select_representative_block(
    test_blocks: dict[str, pathlib.Path],
    label_key: str = 'label',
) -> tuple[str, pathlib.Path]:
    '''
    Deterministically select a representative block with class diversity.

    Evaluates candidate blocks to find one containing a non-trivial
    mixture of classes, avoiding blank or homogeneous background blocks.

    Args:
        test_blocks:
            Dictionary mapping block name to filesystem path.
        label_key:
            Name of label array stored in `.npz` files.

    Returns:
        tuple[str, pathlib.Path]:
            Selected block name and file path.

    Raises:
        ValueError:
            If `test_blocks` is empty.
    '''
    if not test_blocks:
        raise ValueError('No test blocks available to select from.')
    if len(test_blocks) == 1:
        return next(iter(test_blocks.items()))

    best_name: str | None = None
    best_path: pathlib.Path | None = None
    best_score = -1

    # check candidate blocks (up to 20 to preserve fast I/O)
    sorted_items = sorted(test_blocks.items())
    for name, path in sorted_items[:20]:
        score = _score_block_diversity(path, label_key)
        if score is not None and score > best_score:
            best_score = score
            best_name = name
            best_path = path

    if best_name is not None and best_path is not None:
        return best_name, best_path

    # fallback to middle candidate
    return sorted_items[len(sorted_items) // 2]


def run_preview_inference(
    *,
    checkpoint_fpath: str | pathlib.Path,
    dataspecs: core.DataSpecs,
    model: core.MultiheadModelLike,
    config: factory.SessionConfigShape,
    device: str = 'auto',
    active_heads: list[str] | None = None,
    logger: session_logger.SessionLogger | None = None,
) -> core.InferStepResults | None:
    '''
    Load checkpoint weights and run stitched test preview inference.

    Executes evaluation over the full contiguous test mosaic when grid
    dimensions match `dataspecs.splits.test`, or falls back to a single
    representative diverse test block otherwise.

    Args:
        checkpoint_fpath:
            Path to saved model checkpoint weights (`.pt`).
        dataspecs:
            Data specification object containing splits and metadata.
        model:
            Assembled multi-head model instance.
        config:
            Session configuration supplying dataloader and engine setup.
        device:
            Target execution device (e.g. `'cuda'`, `'cpu'`, `'auto'`).
        active_heads:
            Optional subset of heads to activate; defaults to all heads.
        logger:
            Optional session logger for status messages.

    Returns:
        core.InferStepResults | None:
            Stitched inputs, labels, predictions, and errors, or `None`
            when no test blocks or checkpoint exist.
    '''
    if not dataspecs.splits.test:
        if logger:
            logger.info(
                'No test blocks found in dataspecs; skipping preview '
                'inference.'
            )
        return None

    ckpt_path = pathlib.Path(checkpoint_fpath)
    if not ckpt_path.is_file():
        if logger:
            logger.warning(f'Checkpoint file not found: {ckpt_path}')
        return None

    target_device = runtime.resolve_device(device)
    artifacts.load_checkpoint(
        model=model,
        fpath=ckpt_path,
        map_device=target_device,
    )

    preview_specs = _resolve_preview_dataspecs(dataspecs)
    dataloaders = data.build_dataloaders(
        preview_specs,
        config.dataloader,
        logger=None,
    )
    dispatcher = instrumentation.build_dispatcher(logger=logger)
    epoch_runner = engine_mod.build_engine(
        engine_mod.EngineContext(
            dataspecs=preview_specs,
            model=model,
            dataloaders=dataloaders,
            dispatcher=dispatcher,
        ),
        config.engine,
        mode='eval_only',
        eval_dataset='test',
        device=target_device,
    )

    assert epoch_runner.evaluator is not None
    epoch_runner.set_head_state(active_heads=active_heads)
    try:
        return epoch_runner.evaluator.infer(0)
    finally:
        epoch_runner.reset_head_state()


# ----- private helpers
def _score_block_diversity(path: pathlib.Path, label_key: str) -> int | None:
    '''Compute deterministic class-diversity score for a `.npz` block.'''
    try:
        with numpy.load(path, allow_pickle=True) as loaded:
            if label_key not in loaded:
                return None
            lbl = loaded[label_key]
            unique_classes = len(numpy.unique(lbl))
            valid_px = int(numpy.count_nonzero(lbl))
            return unique_classes * 1_000_000 + valid_px
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _resolve_preview_dataspecs(dataspecs: core.DataSpecs) -> core.DataSpecs:
    '''Resolve `test_only` specs for full mosaic or fallback block.'''
    test_blocks = dataspecs.splits.test
    grid_col, grid_row = dataspecs.meta.test_blks_grid
    is_contiguous = (
        grid_col > 0
        and grid_row > 0
        and (grid_col * grid_row == len(test_blocks))
    )
    if is_contiguous:
        return dataclasses.replace(dataspecs, mode='test_only')

    rep_name, rep_path = select_representative_block(
        test_blocks,
        label_key=dataspecs.meta.label_specs.array_key,
    )
    preview_meta = dataclasses.replace(
        dataspecs.meta,
        test_blks_grid=(1, 1),
    )
    preview_splits = dataclasses.replace(
        dataspecs.splits,
        test={rep_name: rep_path},
    )
    return dataclasses.replace(
        dataspecs,
        mode='test_only',
        meta=preview_meta,
        splits=preview_splits,
    )
