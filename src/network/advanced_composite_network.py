# src/network/advanced_composite_network.py

import torch
import torch.nn as nn
from basecat.layers import LinearLayer, ReLUActivation

class AdvancedCompositeNetwork(nn.Module):
    """
    An advanced composite network that includes:
      - A main branch: linear layer -> ReLU -> linear layer.
      - A secondary branch: a direct linear mapping from input to output.
      - The final output is the sum of the outputs from both branches.
    """
    def __init__(self, in_features=3, hidden_features=4, out_features=2):
        super(AdvancedCompositeNetwork, self).__init__()
        self.layer1 = LinearLayer(in_features, hidden_features)
        self.activation = ReLUActivation()
        self.layer2 = LinearLayer(hidden_features, out_features)
        
        # Secondary branch: direct mapping from input to output.
        self.branch = LinearLayer(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.layer1(x)
        main = self.activation(main)
        main = self.layer2(main)
        
        branch = self.branch(x)
        return main + branch

if __name__ == "__main__":
    net = AdvancedCompositeNetwork()
    x = torch.randn(5, 3)
    y = net(x)
    print("Advanced composite network output:")
    print(y)
