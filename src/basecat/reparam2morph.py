# src/basecat/reparam2morph.py

from typing import Any, Callable
from basecat.morphisms import ParametricMorphism

class Reparam2Morphism:
    """
    Represents a 2-morphism (reparameterization) between two parametric morphisms.
    
    Given:
      - A source morphism: (P, f): X -> Y
      - A target morphism: (P', f'): X -> Y
    and a reparameterization map r: P' -> P, this class ensures that:
    
        f'(p', x) == f(r(p'), x)
    
    for all valid parameters p' and inputs x.
    """
    def __init__(self, 
                 source: ParametricMorphism, 
                 target: ParametricMorphism, 
                 reparam_map: Callable[[Any], Any],
                 name: str = None):
        # Verify that the domain and codomain of source and target are compatible.
        if not (source.dom.is_compatible(target.dom) and source.cod.is_compatible(target.cod)):
            raise ValueError("Source and target morphisms must have compatible domains and codomains.")
        self.source = source
        self.target = target
        self.reparam_map = reparam_map
        self.name = name if name is not None else "UnnamedReparam"

    def apply_on_params(self, p_prime: Any) -> Any:
        """
        Applies the reparameterization map r: P' -> P on a given parameter p_prime.
        """
        return self.reparam_map(p_prime)

    def check_commute(self, test_samples: int = 5, tol: float = 1e-6, use_torch: bool = False) -> bool:
        """
        Verify that for a batch of test samples the following holds:
            f'(p', x) == f(r(p'), x)
        for the reparameterization.
        
        Args:
            test_samples (int): Number of samples to test.
            tol (float): Tolerance for numeric equality.
            use_torch (bool): If True, use torch tensors for sampling; otherwise, use NumPy arrays.
        
        Returns:
            bool: True if the condition holds for all test samples, False otherwise.
        """
        if use_torch:
            import torch
            # Create a batch of inputs using torch.rand.
            if hasattr(self.source.dom, 'shape') and self.source.dom.shape:
                batch_shape = (test_samples,) + self.source.dom.shape
                X = torch.rand(batch_shape)
            else:
                X = torch.rand(test_samples)
        else:
            import numpy as np
            # Create a batch of inputs using np.random.rand.
            if hasattr(self.source.dom, 'shape') and self.source.dom.shape:
                batch_shape = (test_samples,) + self.source.dom.shape
                X = np.random.rand(*batch_shape)
            else:
                X = np.random.rand(test_samples)
        
        for i in range(test_samples):
            x = X[i] if test_samples > 1 else X
            if use_torch:
                p_prime = torch.rand(1)
                y_target = self.target.forward(p_prime, x)
                p = self.reparam_map(p_prime)
                y_source = self.source.forward(p, x)
                if not torch.allclose(y_target, y_source, atol=tol):
                    print(f"Sample {i} failed: torch.allclose condition not met.")
                    return False
            else:
                p_prime = np.random.rand(1)
                y_target = self.target.forward(p_prime, x)
                p = self.reparam_map(p_prime)
                y_source = self.source.forward(p, x)
                if not np.allclose(y_target, y_source, atol=tol):
                    print(f"Sample {i} failed: np.allclose condition not met.")
                    return False
        return True

    def __str__(self):
        return f"Reparam2Morphism(name={self.name}, source={self.source.name}, target={self.target.name})"
    
    def __repr__(self):
        return self.__str__()
