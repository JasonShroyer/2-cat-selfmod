# src/basecat/layers.py

import torch
import torch.nn as nn
from basecat.objects import CatObject, TupleParamSpace
from basecat.diff_morphisms import DifferentiableMorphism

class LinearLayer(nn.Module):
    """
    A simple linear layer with bias, implemented as a differentiable morphism.
    Computes y = x @ W^T + b.
    """
    def __init__(self, in_features: int, out_features: int):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Define domain and codomain objects.
        self.dom = CatObject("Input", shape=(in_features,))
        self.cod = CatObject("Output", shape=(out_features,))
        
        # Initialize weight and bias as trainable parameters.
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

# --- Additional Activation Layers ---

class SigmoidActivation(nn.Module):
    """
    A sigmoid activation function.
    """
    def __init__(self):
        super(SigmoidActivation, self).__init__()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(x)

class TanhActivation(nn.Module):
    """
    A tanh activation function.
    """
    def __init__(self):
        super(TanhActivation, self).__init__()
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh(x)

class LeakyReLUActivation(nn.Module):
    """
    A LeakyReLU activation function.
    """
    def __init__(self, negative_slope: float = 0.01):
        super(LeakyReLUActivation, self).__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu(x)

# --- Advanced Linear Layer with Dropout Support ---

class AdvancedLinearLayer(nn.Module):
    """
    An advanced linear layer that includes dropout.
    Computes y = x @ W^T + b, then applies dropout.
    """
    def __init__(self, in_features: int, out_features: int, dropout_prob: float = 0.5):
        super(AdvancedLinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_prob = dropout_prob
        
        self.dom = CatObject("Input", shape=(in_features,))
        self.cod = CatObject("Output", shape=(out_features,))
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # Use PyTorch's dropout layer.
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.addmm(self.bias, x, self.weight.t())
        return self.dropout(x)
