# src/self_modification.py

import copy
import torch
import torch.nn as nn
from project_mnist import MNISTClassifier  # Adjust if necessary

def reparameterize_fc1(old_layer: nn.Module, scale: float) -> (nn.Module, callable):
    """
    Given an old fully-connected layer (old_layer) that acts as fc1,
    create a new layer where the parameters are scaled by 1/scale,
    and adjust the forward pass so that the overall computation remains unchanged.

    For a linear layer (y = x @ W^T + b), if we set:
        W_new = W_old / scale   and   b_new = b_old / scale,
    then the new layer's computation is:
        y_new = x @ (W_old/scale)^T + (b_old/scale) = (1/scale) * (x @ W_old^T + b_old)

    To compensate, we override the forward pass so that:
        new_layer(x) = scale * y_new = x @ W_old^T + b_old

    The reparameterization map r is defined as:
        r(p_new) = scale * p_new,
    which maps new parameters back to the old scale.

    Returns:
        new_layer: a new linear layer with the adjusted forward pass.
        r: a reparameterization function that maps new parameters to the old scale.
    """
    # Extract dimensions from the old layer.
    in_features = old_layer.in_features
    out_features = old_layer.out_features
    
    # Import our custom LinearLayer from basecat.layers.
    from basecat.layers import LinearLayer
    new_layer = LinearLayer(in_features, out_features)
    
    # Set new parameters to scaled versions of the old parameters.
    with torch.no_grad():
        new_layer.weight.copy_(old_layer.weight / scale)
        new_layer.bias.copy_(old_layer.bias / scale)
    
    # Override the forward pass: multiply the result by scale.
    original_forward = new_layer.forward
    def new_forward(x):
        return scale * original_forward(x)
    new_layer.forward = new_forward
    
    # Define the reparameterization map.
    def reparam_map(new_param):
        return scale * new_param
    
    return new_layer, reparam_map

def self_modify_fc1(model: MNISTClassifier, scale: float = 2.0) -> (MNISTClassifier, callable):
    """
    Self-modify the MNISTClassifier by reparameterizing its fc1 layer.
    
    This function:
      - Creates a deep copy of the original model.
      - Replaces the fc1 layer with a new version whose parameters are scaled by 1/scale,
        with an adjusted forward pass that compensates by multiplying by scale.
      - Returns the new model along with a reparameterization function r for fc1 parameters.
    
    Args:
        model (MNISTClassifier): The original MNIST classifier model.
        scale (float): The scaling factor for reparameterization.
        
    Returns:
        new_model (MNISTClassifier): The modified model with the new fc1.
        r (callable): The reparameterization function for fc1 parameters.
    """
    # Deep copy the original model so that we do not modify it.
    new_model = copy.deepcopy(model)
    
    # Retrieve the original fc1 layer.
    old_fc1 = model.fc1
    # Reparameterize fc1: create a new fc1 and get the reparameterization map.
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
      - Computes the output of the modified model.
      - Verifies that the outputs are equal (within a numerical tolerance).
    """
    from project_mnist import MNISTClassifier
    
    # Instantiate the original model.
    original_model = MNISTClassifier()
    original_model.eval()
    
    # Create a dummy input: shape (batch, channels, height, width) = (1, 1, 28, 28)
    x = torch.randn(1, 1, 28, 28)
    
    with torch.no_grad():
        original_output = original_model(x)
    
    # Apply self-modification to fc1.
    modified_model, r = self_modify_fc1(original_model, scale=2.0)
    modified_model.eval()
    
    with torch.no_grad():
        modified_output = modified_model(x)
    
    # Compare the outputs.
    if torch.allclose(original_output, modified_output, atol=1e-6):
        print("Self-modification preserved model output!")
    else:
        print("Self-modification FAILED: outputs differ.")
        print("Original output:", original_output)
        print("Modified output:", modified_output)

if __name__ == "__main__":
    test_self_modification()
