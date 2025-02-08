# tests/test_composite_network.py

import torch
import pytest
from network.composite_network import CompositeNetwork

def test_composite_forward():
    net = CompositeNetwork()
    x = torch.tensor([1.0])
    # Expected output: 1.0 * 2.0 * 3.0 = 6.0
    y = net(x)
    assert torch.allclose(y, torch.tensor([6.0])), "Composite forward pass failed."

def test_composite_backward():
    net = CompositeNetwork()
    x = torch.tensor([1.0], requires_grad=True)
    y = net(x)
    loss = y.sum()
    loss.backward()
    # Check that gradients are computed for both layers.
    assert net.layer1.param.grad is not None, "Layer1 gradient is None"
    assert net.layer2.param.grad is not None, "Layer2 gradient is None"

if __name__ == "__main__":
    pytest.main()
