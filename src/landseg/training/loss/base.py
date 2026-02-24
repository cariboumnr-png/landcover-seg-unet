'''Base classes for loss component.'''

# standard imports
import abc
# third-party imports
import torch
import torch.nn

class PrimitiveLoss(torch.nn.Module, metaclass=abc.ABCMeta):
    '''Base class for loss primitives.'''
    @abc.abstractmethod
    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            *,
            masks: dict[float, torch.Tensor] | None,
        ) -> torch.Tensor:
        '''Forward.'''
