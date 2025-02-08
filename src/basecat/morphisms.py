# src/basecat/morphisms.py

from typing import Callable, Any
from basecat.objects import CatObject, MonoidalObject, TupleParamSpace

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
        self.dom = dom      # Domain (input space)
        self.cod = cod      # Codomain (output space)
        self.param_obj = param_obj  # Parameter space (e.g., TupleParamSpace)
        self.apply_fn = apply_fn    # Function f: (theta, x) -> y
        self.name = name if name is not None else "UnnamedMorphism"

    def forward(self, theta: Any, x: Any) -> Any:
        """
        Compute f(theta, x).
        """
        return self.apply_fn(theta, x)

    def compose(self, other: 'ParametricMorphism') -> 'ParametricMorphism':
        """
        Compose this morphism with another.
        If self: X -> Y and other: Y -> Z, then the composition is:
            other ∘ self: X -> Z.
        The new parameter space is the product of self.param_obj and other.param_obj.
        """
        if not self.cod.is_compatible(other.dom):
            raise ValueError(f"Cannot compose: codomain {self.cod} not compatible with domain {other.dom}")

        # Combine parameter spaces via their product method.
        new_param_obj = self.param_obj.product(other.param_obj)
        
        def composed_apply_fn(params, x):
            # We assume params is a tuple: (params_self, params_other).
            # In practice, you might want a more robust splitting.
            params_self = params[0]
            params_other = params[1]
            intermediate = self.apply_fn(params_self, x)
            return other.apply_fn(params_other, intermediate)
        
        new_name = f"({other.name} ∘ {self.name})"
        return ParametricMorphism(self.dom, other.cod, new_param_obj, composed_apply_fn, name=new_name)

    def __str__(self):
        return f"ParametricMorphism(name={self.name}, dom={self.dom}, cod={self.cod})"

    def __repr__(self):
        return self.__str__()
