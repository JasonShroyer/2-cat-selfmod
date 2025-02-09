# tests/test_layers_extra.py

import torch
import pytest
from basecat.layers import SigmoidActivation, TanhActivation, LeakyReLUActivation, AdvancedLinearLayer

def test_sigmoid_activation():
    sigmoid = SigmoidActivation()
    x = torch.tensor([[-1.0, 0.0, 1.0]])
    y = sigmoid(x)
    expected = torch.sigmoid(x)
    assert torch.allclose(y, expected), "SigmoidActivation did not match expected output."

def test_tanh_activation():
    tanh = TanhActivation()
    x = torch.tensor([[-1.0, 0.0, 1.0]])
    y = tanh(x)
    expected = torch.tanh(x)
    assert torch.allclose(y, expected), "TanhActivation did not match expected output."

def test_leaky_relu_activation():
    negative_slope = 0.1
    leaky_relu = LeakyReLUActivation(negative_slope=negative_slope)
    x = torch.tensor([[-1.0, 0.0, 1.0]])
    y = leaky_relu(x)
    expected = torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)
    assert torch.allclose(y, expected), "LeakyReLUActivation did not match expected output."

def test_advanced_linear_layer():
    # Test that AdvancedLinearLayer produces output of the expected shape.
    layer = AdvancedLinearLayer(in_features=4, out_features=3, dropout_prob=0.0)  # set dropout to 0 for deterministic behavior.
    x = torch.randn(5, 4)  # Batch of 5, input dim 4.
    y = layer(x)
    assert y.shape == (5, 3), "AdvancedLinearLayer output shape mismatch."

if __name__ == "__main__":
    pytest.main()
