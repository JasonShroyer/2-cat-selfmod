# src/self_modification.py

import copy
import torch
import torch.nn as nn
from project_mnist import MNISTClassifier  # Adjust if necessary

def reparameterize_fc1(old_layer: nn.Module, scale: float) -> (nn.Module, callable):
    """
    Reparameterize the fc1 layer by scaling its base parameters.
    
    If old_layer already has attributes `base_weight` and `base_bias`, these are used
    as the original parameters; otherwise, the current weight and bias are stored as the base.
    
    The new layer is constructed with:
      - new_weight = base_weight / scale
      - new_bias = base_bias / scale
    And the forward pass is overridden so that:
      new_layer(x) = scale * (x @ (base_weight/scale)^T + (base_bias/scale))
                    = x @ base_weight^T + base_bias
    The reparameterization map is defined as:
      r(p_new) = scale * p_new
      
    This ensures that regardless of the scale applied in this modification,
    the overall computation remains the same as the original layer.
    
    Returns:
        new_layer: The new fc1 layer with adjusted forward pass.
        r: A reparameterization function mapping new parameters to the original scale.
    """
    # If the old layer does not have base parameters, store them.
    if not hasattr(old_layer, 'base_weight'):
        old_layer.base_weight = old_layer.weight.clone().detach()
        old_layer.base_bias = old_layer.bias.clone().detach()
    
    # Use the base parameters for further modifications.
    base_weight = old_layer.base_weight
    base_bias = old_layer.base_bias
    in_features = old_layer.in_features
    out_features = old_layer.out_features
    
    from basecat.layers import LinearLayer  # Import our custom linear layer.
    new_layer = LinearLayer(in_features, out_features)
    
    with torch.no_grad():
        # Set new layer's parameters based on the base parameters scaled by 1/scale.
        new_layer.weight.copy_(base_weight / scale)
        new_layer.bias.copy_(base_bias / scale)
    
    # Store base parameters in the new layer for future modifications.
    new_layer.base_weight = base_weight
    new_layer.base_bias = base_bias
    
    # Capture the original forward function of new_layer.
    original_forward = new_layer.forward
    
    # Override the forward pass so that the layer output compensates for the scaling.
    def new_forward(x):
        return scale * original_forward(x)
    
    new_layer.forward = new_forward
    
    # Define the reparameterization map: r(p_new) = scale * p_new.
    def reparam_map(new_param):
        return scale * new_param
    
    return new_layer, reparam_map

def self_modify_fc1(model: MNISTClassifier, scale: float = 2.0) -> (MNISTClassifier, callable):
    """
    Self-modify the MNISTClassifier by reparameterizing its fc1 layer.
    
    This function creates a deep copy of the model, then replaces its fc1 layer with a new version
    obtained via reparameterize_fc1. The new fc1 uses the original (base) parameters for computation,
    ensuring that repeated modifications are idempotent.
    
    Args:
        model (MNISTClassifier): The original MNIST classifier.
        scale (float): The scaling factor for this modification.
    
    Returns:
        new_model (MNISTClassifier): The modified model with updated fc1.
        r (callable): The reparameterization function for fc1 parameters.
    """
    new_model = copy.deepcopy(model)
    old_fc1 = model.fc1
    new_fc1, r = reparameterize_fc1(old_fc1, scale)
    new_model.fc1 = new_fc1
    return new_model, r

def test_self_modification():
    """
    Test that self-modification preserves the model's output.
    
    This test:
      - Instantiates an original MNISTClassifier.
      - Computes its output on a dummy input.
      - Applies self-modification to fc1.
      - Verifies that the modified model produces the same output as the original.
    """
    from project_mnist import MNISTClassifier
    
    original_model = MNISTClassifier()
    original_model.eval()
    
    x = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        orig_output = original_model(x)
    
    modified_model, r = self_modify_fc1(original_model, scale=2.0)
    modified_model.eval()
    with torch.no_grad():
        mod_output = modified_model(x)
    
    if torch.allclose(orig_output, mod_output, atol=1e-6):
        print("Self-modification preserved model output!")
    else:
        print("Self-modification FAILED: outputs differ.")
        print("Original output:", orig_output)
        print("Modified output:", mod_output)

if __name__ == "__main__":
    test_self_modification()
