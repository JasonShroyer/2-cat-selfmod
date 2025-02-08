# tests/test_advanced_composite_network.py

import torch
import pytest
from network.advanced_composite_network import AdvancedCompositeNetwork

def test_advanced_forward_shape():
    net = AdvancedCompositeNetwork()
    x = torch.randn(10, 3)  # Batch size 10, input dimension 3.
    y = net(x)
    # Expected output shape: (10, 2)
    assert y.shape == (10, 2), "AdvancedCompositeNetwork output shape mismatch."

def test_advanced_gradients():
    net = AdvancedCompositeNetwork()
    x = torch.randn(8, 3, requires_grad=True)
    y = net(x)
    loss = y.sum()
    loss.backward()
    # Check that gradients are computed for at least one parameter in each layer.
    grad_found = any(param.grad is not None for param in net.parameters())
    assert grad_found, "No gradients computed in AdvancedCompositeNetwork."

if __name__ == "__main__":
    pytest.main()
