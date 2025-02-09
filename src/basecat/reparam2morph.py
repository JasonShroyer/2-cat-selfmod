# src/basecat/reparam2morph.py

from typing import Any, Callable
from basecat.morphisms import ParametricMorphism
import numpy as np
import torch

class Reparam2Morphism:
    """
    Represents a 2-morphism (reparameterization) between two parametric morphisms.
    
    Given a source morphism (P, f): X -> Y and a target morphism (P', f'): X -> Y,
    along with a reparameterization map r: P' -> P, this class ensures that:
    
        f'(p', x) == f(r(p'), x)
    
    for all valid parameters p' and inputs x.
    """
    def __init__(self, 
                 source: ParametricMorphism, 
                 target: ParametricMorphism, 
                 reparam_map: Callable[[Any], Any],
                 name: str = None):
        if not (source.dom.is_compatible(target.dom) and source.cod.is_compatible(target.cod)):
            raise ValueError("Source and target morphisms must have compatible domains and codomains.")
        self.source = source
        self.target = target
        self.reparam_map = reparam_map
        self.name = name if name is not None else "UnnamedReparam"
    
    def apply_on_params(self, p_prime: Any) -> Any:
        return self.reparam_map(p_prime)
    
    def check_commute(self, test_samples: int = 5, tol: float = 1e-6, use_torch: bool = False) -> bool:
        """
        Verify that for a set of test samples, the condition
            f'(p', x) == f(r(p'), x)
        holds.
        
        If use_torch is True, uses a vectorized Torch-based approach; otherwise, uses NumPy.
        """
        if use_torch:
            return self.check_commute_vectorized(tol=tol)
        else:
            if hasattr(self.source.dom, 'shape') and self.source.dom.shape:
                batch_shape = (test_samples,) + self.source.dom.shape
                X = np.random.rand(*batch_shape)
            else:
                X = np.random.rand(test_samples)
            for i in range(test_samples):
                x = X[i] if test_samples > 1 else X
                p_prime = np.random.rand(1)  # For simplicity, assume scalar parameter.
                y_target = self.target.forward(p_prime, x)
                p = self.reparam_map(p_prime)
                y_source = self.source.forward(p, x)
                if not np.allclose(y_target, y_source, atol=tol):
                    print(f"Sample {i} failed: np.allclose condition not met.")
                    return False
            return True
    
    def check_commute_vectorized(self, tol: float = 1e-6) -> bool:
        """
        A vectorized implementation of check_commute using Torch tensors.
        
        This method samples a batch of inputs and a corresponding batch of parameters,
        then computes the target and source outputs in a loop and stacks the results.
        Finally, it checks that the outputs are equal (within the given tolerance) across the batch.
        """
        test_samples = 10  # You can adjust the number of samples here.
        # Generate a batch of inputs.
        if hasattr(self.source.dom, 'shape') and self.source.dom.shape:
            batch_shape = (test_samples,) + self.source.dom.shape
            X = torch.rand(batch_shape)
        else:
            X = torch.rand(test_samples)
        # Generate a batch of parameters (assume scalar for simplicity).
        p_primes = torch.rand(test_samples, 1)
        
        y_targets = []
        y_sources = []
        for i in range(test_samples):
            x = X[i]
            p_prime = p_primes[i]
            y_target = self.target.forward(p_prime, x)
            p = self.reparam_map(p_prime)
            y_source = self.source.forward(p, x)
            y_targets.append(y_target)
            y_sources.append(y_source)
        
        y_targets = torch.stack(y_targets)
        y_sources = torch.stack(y_sources)
        if torch.allclose(y_targets, y_sources, atol=tol):
            return True
        else:
            print("Vectorized check failed:")
            print("y_targets:", y_targets)
            print("y_sources:", y_sources)
            return False
    
    def __str__(self):
        return f"Reparam2Morphism(name={self.name}, source={self.source.name}, target={self.target.name})"
    
    def __repr__(self):
        return self.__str__()

