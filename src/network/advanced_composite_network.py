# src/network/advanced_composite_network.py

import torch
import torch.nn as nn
from basecat.layers import LinearLayer, ReLUActivation

class AdvancedCompositeNetwork(nn.Module):
    """
    An advanced composite network that includes:
      - Two linear layers with bias.
      - A ReLU activation between them.
      - A branch: one branch performs an additional linear transformation,
        and the output is the sum of the two branches.
    """
    def __init__(self, in_features=3, hidden_features=4, out_features=2):
        super(AdvancedCompositeNetwork, self).__init__()
        self.layer1 = LinearLayer(in_features, hidden_features)
        self.activation = ReLUActivation()
        self.layer2 = LinearLayer(hidden_features, out_features)
        
        # Additional branch: a simple linear layer mapping input directly to output.
        self.branch = LinearLayer(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main path: x -> layer1 -> ReLU -> layer2.
        main = self.layer1(x)
        main = self.activation(main)
        main = self.layer2(main)
        
        # Branch: direct mapping from input to output.
        branch = self.branch(x)
        
        # Sum the outputs.
        return main + branch

if __name__ == "__main__":
    net = AdvancedCompositeNetwork()
    x = torch.randn(5, 3)  # Batch of 5 samples, 3 features each.
    y = net(x)
    print("Advanced composite network output:")
    print(y)
