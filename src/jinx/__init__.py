""" Jinx. """

from typing import NamedTuple, Protocol, Self
from torch import Tensor

class Parameters(NamedTuple):
    """ Parameters of the model. """
    ...

class Model(Protocol):
    """ Define required methods for an online variational inference model. """
    def __init__(self: Self) -> None: ...
    def update(self: Self, X: Tensor, y: Tensor) -> None: ...
    def weight_posterior(self: Self) -> dict[str, Tensor]: ...
