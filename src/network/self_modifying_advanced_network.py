# src/network/self_modifying_advanced_network.py

import copy
import torch
from advanced_composite_network import AdvancedCompositeNetwork
from self_modification import reparameterize_fc1

def self_modify_layer1(model: AdvancedCompositeNetwork, scale: float = 2.0):
    """
    Self-modify the 'layer1' of an AdvancedCompositeNetwork by reparameterizing it.
    
    For a given linear layer (y = x @ W^T + b), we define:
      - new_weight = base_weight / scale
      - new_bias   = base_bias / scale
    and override the forward pass so that:
      new_layer(x) = scale * (x @ (base_weight/scale)^T + (base_bias/scale))
                   = x @ base_weight^T + base_bias
    The function returns the modified model and the reparameterization map r,
    defined as r(p_new) = scale * p_new.
    
    This ensures that if the original layer produces output y,
    then after modification, the new layer produces the same output.
    
    Args:
        model (AdvancedCompositeNetwork): The original advanced composite network.
        scale (float): The scaling factor for the modification.
        
    Returns:
        new_model (AdvancedCompositeNetwork): The modified network with updated layer1.
        r (callable): The reparameterization function.
    """
    # Deep copy the model to avoid modifying the original.
    new_model = copy.deepcopy(model)
    
    # Retrieve the original 'layer1'. We assume it is an instance of our custom LinearLayer.
    old_layer = model.layer1
    
    # Use our reparameterization function to create a new layer.
    new_layer, r = reparameterize_fc1(old_layer, scale)
    
    # Replace layer1 in the copied model.
    new_model.layer1 = new_layer
    
    return new_model, r

def test_self_modification_advanced():
    """
    Test that self-modification of layer1 in the AdvancedCompositeNetwork preserves output.
    
    Steps:
      1. Instantiate an AdvancedCompositeNetwork.
      2. Compute its output on a dummy input.
      3. Apply self-modification to layer1 (with a chosen scale factor).
      4. Compute the output of the modified network.
      5. Compare outputs: they should be equal (within tolerance).
    """
    # Instantiate the advanced composite network.
    model = AdvancedCompositeNetwork()
    model.eval()
    
    # Create a dummy input. For AdvancedCompositeNetwork, input shape is (batch, in_features),
    # where in_features is defined in its constructor (default 3).
    x = torch.randn(1, 3)
    
    with torch.no_grad():
        original_output = model(x)
    
    # Apply self-modification to layer1 using a scale factor of 2.0.
    modified_model, r = self_modify_layer1(model, scale=2.0)
    modified_model.eval()
    
    with torch.no_grad():
        modified_output = modified_model(x)
    
    if torch.allclose(original_output, modified_output, atol=1e-6):
        print("Advanced network self-modification preserved output!")
    else:
        print("Advanced network self-modification FAILED: outputs differ.")
        print("Original output:", original_output)
        print("Modified output:", modified_output)

if __name__ == "__main__":
    test_self_modification_advanced()
