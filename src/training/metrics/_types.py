'''doc'''

# standard imports
import typing

class ConfusionMatrixConfig(typing.TypedDict):
    '''Config dict for confusion matrix compute.'''
    num_classes: int
    ignore_index: int
    parent_class_1b: int | None
    exclude_class_1b: tuple[int, ...] | None
