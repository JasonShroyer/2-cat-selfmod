# src/self_modification_multi_layer.py

import copy
import torch
import torch.nn as nn
from project_mnist import MNISTClassifier  # Adjust if your MNISTClassifier is defined elsewhere

def reparameterize_layer(layer: nn.Module, scale: float) -> (nn.Module, callable):
    """
    Generic reparameterization for a linear layer (assumed to have 'weight' and 'bias' attributes).
    
    If the layer does not have base parameters (base_weight, base_bias), these are stored from the
    current parameters. Then, a new layer of the same type is created where:
    
        new_weight = base_weight / scale
        new_bias   = base_bias / scale
    
    The forward pass is overridden to multiply the result by 'scale', ensuring that:
    
        new_layer(x) = scale * (x @ (base_weight/scale)^T + (base_bias/scale))
                      = x @ base_weight^T + base_bias
    
    This guarantees that the layer's overall computation is preserved.
    
    The reparameterization map is defined as:
        r(p_new) = scale * p_new,
    mapping new parameters to the original scale.
    
    Args:
        layer: The linear layer to reparameterize.
        scale: The scaling factor to use for modification.
        
    Returns:
        new_layer: A new layer instance with adjusted parameters and forward pass.
        r: A reparameterization function (callable) for the layer's parameters.
    """
    # If base parameters are not already stored, save them.
    if not hasattr(layer, 'base_weight'):
        layer.base_weight = layer.weight.clone().detach()
        layer.base_bias = layer.bias.clone().detach()
    
    base_weight = layer.base_weight
    base_bias = layer.base_bias
    in_features = layer.in_features
    out_features = layer.out_features
    
    # Import our custom LinearLayer from basecat.layers.
    from basecat.layers import LinearLayer
    new_layer = LinearLayer(in_features, out_features)
    
    with torch.no_grad():
        new_layer.weight.copy_(base_weight / scale)
        new_layer.bias.copy_(base_bias / scale)
    
    # Store the base parameters in the new layer.
    new_layer.base_weight = base_weight
    new_layer.base_bias = base_bias
    
    # Capture the original forward function.
    original_forward = new_layer.forward
    
    # Override the forward pass so that the output is scaled back.
    def new_forward(x):
        return scale * original_forward(x)
    new_layer.forward = new_forward
    
    # Define the reparameterization map.
    def reparam_map(new_param):
        return scale * new_param
    
    return new_layer, reparam_map

def self_modify_layer(model: nn.Module, layer_name: str, scale: float) -> (nn.Module, callable):
    """
    Generic self-modification function that replaces the layer specified by layer_name in the model
    with a reparameterized version using the provided scaling factor.
    
    Args:
        model: The model containing the layer to modify.
        layer_name: The attribute name of the layer (e.g., "fc1", "fc2").
        scale: The scaling factor for this modification.
    
    Returns:
        new_model: A deep copy of the model with the layer modified.
        r: The reparameterization function for that layer.
    """
    new_model = copy.deepcopy(model)
    old_layer = getattr(new_model, layer_name)
    new_layer, r = reparameterize_layer(old_layer, scale)
    setattr(new_model, layer_name, new_layer)
    return new_model, r

def test_multi_layer_self_modification():
    """
    Test that self-modifying multiple layers of the MNISTClassifier preserves overall output.
    
    In this test:
      - We modify both fc1 and fc2.
      - We apply a scaling factor of 2.0 to fc1 and 0.5 to fc2.
      - Since 2.0 * 0.5 = 1, the net effect should preserve the original model output.
    """
    from project_mnist import MNISTClassifier
    original_model = MNISTClassifier()
    original_model.eval()
    
    # Create a dummy input: MNIST images of shape (batch, 1, 28, 28)
    x = torch.randn(1, 1, 28, 28)
    
    with torch.no_grad():
        orig_output = original_model(x)
    
    # Apply self-modification to fc1 with scale 2.0.
    modified_model, r1 = self_modify_layer(original_model, "fc1", scale=2.0)
    # Then, apply self-modification to fc2 with scale 0.5.
    modified_model, r2 = self_modify_layer(modified_model, "fc2", scale=0.5)
    
    modified_model.eval()
    with torch.no_grad():
        mod_output = modified_model(x)
    
    if torch.allclose(orig_output, mod_output, atol=1e-6):
        print("Multi-layer self-modification preserved model output!")
    else:
        print("Multi-layer self-modification FAILED: outputs differ.")
        print("Original output:", orig_output)
        print("Modified output:", mod_output)

if __name__ == "__main__":
    test_multi_layer_self_modification()
