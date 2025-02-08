# src/basecat/diff_morphisms.py

import torch
import torch.nn as nn
from typing import Any, Callable
from basecat.objects import CatObject, TupleParamSpace
from basecat.morphisms import ParametricMorphism

class DifferentiableMorphism(nn.Module):
    """
    A PyTorch-based implementation of a parametric morphism.
    This class wraps a function f: (theta, x) -> y and registers
    theta as a trainable parameter using nn.Parameter.
    
    Attributes:
        dom (CatObject): The domain (input space).
        cod (CatObject): The codomain (output space).
        apply_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): 
            A function that computes y from (theta, x).
        param (nn.Parameter): The trainable parameter theta.
        name (str): Optional name for the morphism.
    """
    def __init__(self, 
                 dom: CatObject, 
                 cod: CatObject, 
                 apply_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 init_param: torch.Tensor = None,
                 name: str = None):
        super(DifferentiableMorphism, self).__init__()
        self.dom = dom
        self.cod = cod
        self.name = name if name is not None else "UnnamedDifferentiableMorphism"
        # If an initial parameter is not provided, we initialize a scalar parameter.
        if init_param is None:
            init_param = torch.tensor(1.0)
        # Register the parameter as a trainable parameter.
        self.param = nn.Parameter(init_param)
        self.apply_fn = apply_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass by applying apply_fn to the parameter and the input x.
        """
        return self.apply_fn(self.param, x)

    def __str__(self):
        return f"DifferentiableMorphism(name={self.name}, dom={self.dom}, cod={self.cod})"

    def __repr__(self):
        return self.__str__()
