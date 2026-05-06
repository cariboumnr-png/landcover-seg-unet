# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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

'''Multihead loss computation helpers for trainer.'''

# third-party imports
import torch
# local imports
import landseg.session.engine.protocols as protocols

# -------------------------------Public Function-------------------------------
def multihead_loss(
    *,
    multihead_preds: dict[str, torch.Tensor],
    multihead_targets: dict[str, torch.Tensor],
    features: torch.Tensor,
    headspecs: dict[str, protocols.SpecsLike],
    headlosses: dict[str, protocols.CompositeLossLike],
) -> tuple[torch.Tensor, dict[str, float]]:
    '''
    Compose the total loss from per-head losses.

    Steps:
    - For each head:
        - fetch logits and matching target
        - shift target from 1-based to 0-based (keep 255)
        - optionally gate by parent class if hierarchical
        - apply all specified loss functions (weighted sum)

    Returns:
        (total_loss, per_head): scalar total loss and a dict of detached
        per-head loss values for logging.
    '''

    # infer device from preds (assumes all on same device)
    pred_device = next(iter(multihead_preds.values())).device
    # prep outputs
    total = torch.zeros((), device=pred_device)
    per_head: dict[str, float] = {}

    # iterate through multihead prediction dict
    for head_name, head_pred in multihead_preds.items():
        # resolve parent tensor and class if a child head
        parent_tensor: torch.Tensor | None = None
        parent_name = headspecs[head_name].parent_head
        if parent_name is not None:
            parent_tensor = multihead_targets[parent_name]
        # prep target and optional mask tensors per head
        targets_0b, masks = _prep_loss_compute(
            head_target=multihead_targets[head_name],
            head_spec=headspecs[head_name],
            head_loss=headlosses[head_name],
            parent_tensor=parent_tensor,
        )
        # sanity check
        assert head_pred.shape[-2:] == multihead_targets[head_name].shape[-2:]
        # calculate loss
        loss = headlosses[head_name].forward(
            head_pred,
            targets_0b,
            masks=masks,
            features=features
        )
        total += headspecs[head_name].weight * loss
        # per_head losses are detached scalars for logging only
        per_head[head_name] = float(loss.item())
    # NaN check before output
    if not torch.isfinite(total):
        raise RuntimeError('Contains NaN/Inf loss.')
    return total, per_head

# ------------------------------private  function------------------------------
def _prep_loss_compute(
    *,
    head_target: torch.Tensor,
    head_spec: protocols.SpecsLike,
    head_loss: protocols.CompositeLossLike,
    parent_tensor: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[float, torch.Tensor] | None]:
    '''
    Prepare targets and masks for a single head.

    Builds pixel-weight masks from exclusion classes and optional
    parent-child gating, then converts targets to 0-based labels while
    preserving ignore_index.
    '''

    # get mask while raw and parent tensor is still 1-based
    masks = _get_masks(
        raw=head_target,
        masked_cls=head_spec.exclude_cls,       # optional at runtime
        parent_tensor=parent_tensor,            # optional parent-child gating
        parent_cls_1b=head_spec.parent_cls      # 1-based parent class
    )
    # shift batch to 0-based and calc losses
    target_0 = _shift_1_to_0(head_target, head_loss.ignore_index)
    # return
    return target_0, masks

def _get_masks(
    *,
    raw: torch.Tensor,
    masked_cls: tuple[int, ...] | None=None,
    parent_tensor: torch.Tensor | None=None,
    parent_cls_1b: int | None=None
) -> dict[float, torch.Tensor] | None:
    '''
    Construct a mask dictionary for loss weighting.

    Creates:
        - A down-weight mask (e.g., 0.05) for excluded classes.
        - A hard-zero mask (0.0) for pixels outside the parent class
          when parent gating is active.
    '''

    # masks
    masks: dict[float, torch.Tensor] = {}
    # mask for exclusion classes
    if masked_cls is not None:
        masked_cls_tensor = torch.tensor(masked_cls, device=raw.device)
        exclusion_mask = torch.isin(raw, masked_cls_tensor)
        masks[0.05] = exclusion_mask
    # mask for parent-child gating
    if parent_tensor is not None and parent_cls_1b is not None:
        parent_mask = parent_tensor != parent_cls_1b
        masks[0.0] = parent_mask
    # return with default weight
    return masks if masks else None

def _shift_1_to_0(
    target_1: torch.Tensor,
    ignore_idx: int
) -> torch.Tensor:
    '''
    Convert labels from 1..K to 0..K-1 while preserving ignore_index.
    '''

    t = target_1.clone()
    m = t != ignore_idx
    t[m] = t[m] - 1
    return t
