# tests/test_diff_morphisms.py

import torch
import pytest
from basecat.objects import CatObject
from basecat.diff_morphisms import DifferentiableMorphism

def dummy_apply(param: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    A simple dummy function that multiplies the parameter with the input.
    """
    return param * x

def test_forward_and_backward():
    # Create dummy domain and codomain objects.
    X = CatObject("InputSpace", shape=(1,))
    Y = CatObject("OutputSpace", shape=(1,))
    
    # Create a DifferentiableMorphism with the dummy function.
    morph = DifferentiableMorphism(dom=X, cod=Y, apply_fn=dummy_apply, name="TestDiffMorph")
    
    # Create a dummy input tensor with requires_grad=True.
    x = torch.tensor([2.0], requires_grad=True)
    # Perform the forward pass.
    y = morph(x)
    # Compute a simple loss (sum of outputs).
    loss = y.sum()
    # Perform the backward pass.
    loss.backward()
    
    # Check that gradients are computed.
    assert morph.param.grad is not None, "Gradient for parameter not computed."
    assert x.grad is not None, "Gradient for input not computed."

if __name__ == "__main__":
    pytest.main()
