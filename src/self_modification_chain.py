# src/self_modification_chain.py

import copy
import torch
from project_mnist import MNISTClassifier
from self_modification import self_modify_fc1

def chain_modify_fc1(model: MNISTClassifier, scales: list) -> (MNISTClassifier, list):
    """
    Applies a chain of self-modifications to the fc1 layer of the given model.
    
    Args:
        model (MNISTClassifier): The original MNIST classifier model.
        scales (list): A list of scaling factors to apply sequentially.
                       The product of these factors must equal 1 for output preservation.
    
    Returns:
        modified_model (MNISTClassifier): The model after applying all modifications.
        reparam_maps (list): A list of reparameterization functions corresponding to each modification.
    """
    # Verify that the product of scales equals 1.
    prod = 1.0
    for s in scales:
        prod *= s
    if abs(prod - 1.0) > 1e-6:
        raise ValueError("Product of scales must equal 1 for identity preservation. Got: " + str(prod))
    
    modified_model = copy.deepcopy(model)
    reparam_maps = []
    
    for scale in scales:
        modified_model, r = self_modify_fc1(modified_model, scale=scale)
        reparam_maps.append(r)
    
    return modified_model, reparam_maps

def test_chain_modification():
    """
    Test that a chain of self-modifications preserves the original model output.
    
    In this test, we apply two modifications:
      - First with scale = 2.0.
      - Second with scale = 0.5.
    Since 2.0 * 0.5 = 1, the net effect should be the identity.
    """
    from project_mnist import MNISTClassifier
    
    original_model = MNISTClassifier()
    original_model.eval()
    
    x = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        orig_output = original_model(x)
    
    scales = [2.0, 0.5]
    modified_model, reparam_maps = chain_modify_fc1(original_model, scales)
    modified_model.eval()
    
    with torch.no_grad():
        mod_output = modified_model(x)
    
    if torch.allclose(orig_output, mod_output, atol=1e-6):
        print("Chain modification preserved model output!")
    else:
        print("Chain modification FAILED: outputs differ.")
        print("Original output:", orig_output)
        print("Modified output:", mod_output)

if __name__ == "__main__":
    test_chain_modification()
