# src/basecat/morphisms.py

from typing import Callable, Any, Tuple
from basecat.objects import CatObject, MonoidalObject, TupleParamSpace

def split_params(params: Any) -> Tuple[Any, Any]:
    """
    Splits the composite parameter into two parts.
    
    For now, we assume params is a tuple or list with exactly two elements.
    Later, we can extend this to support more complex structures.
    """
    if isinstance(params, (tuple, list)) and len(params) == 2:
        return params[0], params[1]
    else:
        raise ValueError("Expected composite parameters as a tuple or list of length 2.")

def merge_params(p1: Any, p2: Any) -> Any:
    """
    Merges two parameter sets into one composite parameter.
    
    For our TupleParamSpace implementation, this is simply a tuple.
    This function serves as a single point of change if you later want to use a different representation.
    """
    return (p1, p2)

class ParametricMorphism:
    """
    Represents a 1-morphism: (P, f): X -> Y, where:
      - P is the parameter space (an instance of MonoidalObject)
      - f is a function taking (params, x) and returning y.
    """
    def __init__(self, 
                 dom: CatObject, 
                 cod: CatObject, 
                 param_obj: MonoidalObject, 
                 apply_fn: Callable[[Any, Any], Any],
                 name: str = None):
        self.dom = dom
        self.cod = cod
        self.param_obj = param_obj
        self.apply_fn = apply_fn
        self.name = name if name is not None else "UnnamedMorphism"

    def forward(self, theta: Any, x: Any) -> Any:
        return self.apply_fn(theta, x)

    def compose(self, other: 'ParametricMorphism') -> 'ParametricMorphism':
        """
        Compose this morphism with another.
        If self: X -> Y and other: Y -> Z, then composition is:
            other ∘ self: X -> Z.
        """
        if not self.cod.is_compatible(other.dom):
            raise ValueError(f"Cannot compose: codomain {self.cod} not compatible with domain {other.dom}")

        # Combine parameter spaces via their product method.
        new_param_obj = self.param_obj.product(other.param_obj)
        
        def composed_apply_fn(params, x):
            # Use the helper functions to split the composite parameters.
            p1, p2 = split_params(params)
            intermediate = self.apply_fn(p1, x)
            return other.apply_fn(p2, intermediate)
        
        new_name = f"({other.name} ∘ {self.name})"
        return ParametricMorphism(self.dom, other.cod, new_param_obj, composed_apply_fn, name=new_name)

    def __str__(self):
        return f"ParametricMorphism(name={self.name}, dom={self.dom}, cod={self.cod})"

    def __repr__(self):
        return self.__str__()

