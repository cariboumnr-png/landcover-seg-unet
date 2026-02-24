'''Head specifications.'''

# standard imports
import dataclasses

@dataclasses.dataclass
class Spec:
    '''Specifications for a training head.'''
    name: str
    count: list[int]
    loss_alpha: list[float]
    parent_head: str | None
    parent_cls: int | None # 1-based
    weight: float
    exclude_cls: tuple[int, ...]

    def set_exclude(self, indices: tuple[int, ...]) -> None:
        '''Curriculum fills this during training; validates range.'''
        bad = [i for i in indices if i < 1 or i > self.num_classes]
        if bad:
            raise ValueError(f'Invalid indices to exclude: {bad}')
        self.exclude_cls = tuple(sorted(set(indices)))

    @property
    def num_classes(self) -> int:
        '''Number of classes in this head.'''
        return len(self.count)
