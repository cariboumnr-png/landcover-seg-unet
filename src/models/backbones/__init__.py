'''
Top-level namespace for models.backbones.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    #
    'Backbone',
    #
    'DoubleConv',
    'Downsample',
    'Upsample',
    #
    'UNet'
]

# for static check
if typing.TYPE_CHECKING:
    from .base import Backbone
    from .blocks import DoubleConv, Downsample, Upsample
    from .unet import UNet

def __getattr__(name: str):

    if name == 'Backbone':
        return importlib.import_module('.base', __package__).Backbone
    if name == ['DoubleConv', 'Downsample', 'Upsample']:
        return getattr(importlib.import_module('.blocks', __package__), name)
    if name == 'UNet':
        return importlib.import_module('.unet', __package__).UNet

    raise AttributeError(name)
