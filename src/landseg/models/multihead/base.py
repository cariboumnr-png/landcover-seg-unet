'''Base model interface'''

# standard imports
import abc
# third-party imports
import torch
import torch.nn

class BaseMultiheadModel(torch.nn.Module, metaclass=abc.ABCMeta):
    '''With minimal required methods.'''

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        '''Multihead output as a dict of tensors'''

    @abc.abstractmethod
    def set_active_heads(self, active_heads: list[str] | None) -> None:
        '''Heads to be actively trained.'''

    @abc.abstractmethod
    def set_frozen_heads(self, frozen_heads: list[str] | None) -> None:
        '''Heads with no gradient updates.'''

    @abc.abstractmethod
    def reset_heads(self) -> None:
        '''Reset heads state.'''
