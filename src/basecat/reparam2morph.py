# src/basecat/reparam2morph.py

from typing import Any, Callable
from basecat.morphisms import ParametricMorphism

class Reparam2Morphism:
    """
    Represents a 2-morphism (reparameterization) between two parametric morphisms.
    """
    def __init__(self, 
                 source: ParametricMorphism, 
                 target: ParametricMorphism, 
                 reparam_map: Callable[[Any], Any],
                 name: str = None):
        # Ensure that the source and target morphisms have compatible domains and codomains.
        if source.dom != target.dom or source.cod != target.cod:
            raise ValueError("Source and target morphisms must have compatible domains and codomains.")
        self.source = source
        self.target = target
        self.reparam_map = reparam_map
        self.name = name if name is not None else "UnnamedReparam"

    def apply_on_params(self, p_prime: Any) -> Any:
        return self.reparam_map(p_prime)

    def check_commute(self, test_samples: int = 5, tol: float = 1e-6, use_torch: bool = False) -> bool:
        """
        Verify that for a batch of test samples, the condition holds:
             f'(p', x) == f(r(p'), x)
        """
        import numpy as np

        # For simplicity, assume self.source.dom.shape exists.
        if hasattr(self.source.dom, 'shape') and self.source.dom.shape:
            batch_shape = (test_samples,) + self.source.dom.shape
            X = np.random.rand(*batch_shape)
        else:
            X = np.random.rand(test_samples)

        for i in range(test_samples):
            x = X[i]
            p_prime = np.random.rand(1)  # A simple dummy sample.
            y_target = self.target.forward(p_prime, x)
            p = self.reparam_map(p_prime)
            y_source = self.source.forward(p, x)
            if not np.allclose(y_target, y_source, atol=tol):
                print(f"Sample {i} failed: y_target != y_source")
                return False
        return True

    def __str__(self):
        return f"Reparam2Morphism(name={self.name}, source={self.source.name}, target={self.target.name})"

    def __repr__(self):
        return self.__str__()
