# src/basecat/objects.py

from abc import ABC, abstractmethod
from typing import Any, Tuple

class CatObject:
    """
    Represents an object in our base category.
    Typically, this represents a data space (input/output) or a parameter space.
    """
    def __init__(self, name: str, shape: Tuple[int, ...] = (), metadata: dict = None):
        self.name = name
        self.shape = shape  # e.g., (28, 28) for MNIST images or (100,) for a 100-dimensional vector.
        self.metadata = metadata or {}

    def is_compatible(self, other: 'CatObject') -> bool:
        """
        Check if this object is compatible with another (e.g., same shape).
        """
        return self.shape == other.shape

    def __str__(self):
        return f"CatObject(name={self.name}, shape={self.shape})"

    def __repr__(self):
        return self.__str__()

class MonoidalObject(ABC):
    """
    Abstract base class representing a monoidal object.
    This abstracts the idea of a parameter space that can be combined via a product.
    """
    @abstractmethod
    def product(self, other: 'MonoidalObject') -> 'MonoidalObject':
        """
        Combine this object with another to form a product.
        """
        pass

    @staticmethod
    @abstractmethod
    def unit() -> 'MonoidalObject':
        """
        Return the monoidal identity (unit) for parameter spaces.
        """
        pass

class TupleParamSpace(MonoidalObject):
    """
    A concrete implementation of a monoidal object for parameter spaces,
    represented as a tuple of parameter items.
    """
    def __init__(self, params: Tuple[Any, ...]):
        self.params = params

    def product(self, other: 'TupleParamSpace') -> 'TupleParamSpace':
        # Product is simply tuple concatenation.
        return TupleParamSpace(self.params + other.params)

    @staticmethod
    def unit() -> 'TupleParamSpace':
        # The unit is represented as an empty tuple.
        return TupleParamSpace(())

    def __str__(self):
        return f"TupleParamSpace({self.params})"

    def __repr__(self):
        return self.__str__()
