'''Base classes for architecture backbones.'''

# standard imports
import abc
# third-party imports
import torch
import torch.nn

class Backbone(torch.nn.Module, metaclass=abc.ABCMeta):
    '''
    Contract for any feature extractor used by MultiHeadModel.
    Implementations must produce a feature map with a known channel
    width that matches the heads' expected input (e.g., base_ch).
    '''

    @property
    @abc.abstractmethod
    def out_channels(self) -> int:
        '''Number of channels in the output feature map (C_out).'''
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: [B, C_in, H, W]
        Returns:
            y: [B, C_out, H, W] (or possibly downsampled; the framework
                should document expectations, e.g., same spatial size if
                heads are dense).
        '''
        raise NotImplementedError
