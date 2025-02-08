# tests/test_layers.py

import torch
import pytest
from basecat.layers import LinearLayer, ReLUActivation

def test_linear_layer():
    # Create a linear layer.
    layer = LinearLayer(in_features=3, out_features=2)
    # Create a dummy input.
    x = torch.randn(4, 3)  # Batch size 4, 3 features.
    y = layer(x)
    # Check output shape: should be (4, 2)
    assert y.shape == (4, 2), "LinearLayer output shape mismatch."

def test_relu_activation():
    relu = ReLUActivation()
    # Create dummy input.
    x = torch.tensor([[-1.0, 0.0, 2.0]])
    y = relu(x)
    expected = torch.tensor([[0.0, 0.0, 2.0]])
    assert torch.allclose(y, expected), "ReLUActivation did not work as expected."

if __name__ == "__main__":
    pytest.main()
