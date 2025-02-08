# src/basecat/layers.py

import torch
import torch.nn as nn
from basecat.objects import CatObject, TupleParamSpace
from basecat.diff_morphisms import DifferentiableMorphism

class LinearLayer(nn.Module):
    """
    A simple linear layer with bias, implemented as a differentiable morphism.
    Computes y = x * W^T + b.
    """
    def __init__(self, in_features: int, out_features: int):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Define domain and codomain objects (for demonstration, using shapes).
        self.dom = CatObject("Input", shape=(in_features,))
        self.cod = CatObject("Output", shape=(out_features,))
        
        # Initialize weight and bias as trainable parameters.
        # Weight shape: (out_features, in_features)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        # Bias shape: (out_features,)
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear transformation: x @ W^T + b
        return torch.addmm(self.bias, x, self.weight.t())

class ReLUActivation(nn.Module):
    """
    A simple ReLU activation function.
    """
    def __init__(self):
        super(ReLUActivation, self).__init__()
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x)
