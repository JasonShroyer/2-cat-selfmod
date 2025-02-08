# src/basecat/reparam2morph.py

from typing import Callable, Any
from basecat.morphisms import ParametricMorphism

class Reparam2Morphism:
    """
    Represents a 2-morphism (reparameterization) between two parametric morphisms:
      ρ: (P, f) => (P', f'),
    implemented by a map r: P' -> P such that for all x and p' in P':
          f'(p', x) == f(r(p'), x)
    """
    def __init__(self, 
                 source: ParametricMorphism, 
                 target: ParametricMorphism, 
                 reparam_map: Callable[[Any], Any],
                 name: str = None):
        # Ensure the source and target morphisms share the same domain and codomain.
        if not source.dom.is_compatible(target.dom) or not source.cod.is_compatible(target.cod):
            raise ValueError("Source and target morphisms must have compatible domains and codomains.")
        self.source = source
        self.target = target
        self.reparam_map = reparam_map
        self.name = name if name is not None else "UnnamedReparam"

    def apply_on_params(self, p_prime: Any) -> Any:
        """
        Apply the reparameterization map r: P' -> P.
        """
        return self.reparam_map(p_prime)

    def check_commute(self, test_samples: int = 5, tol: float = 1e-6) -> bool:
        """
        Verify the commutativity condition:
           f'(p', x) == f(r(p'), x)
        for a number of test samples.
        
        This is a simplified placeholder version. In practice, you should
        use a sampling method appropriate for your domain.
        """
        import numpy as np

        for i in range(test_samples):
            # Create dummy samples; replace with domain-specific sampling.
            # Assume self.source.dom.shape exists; if not, use a scalar.
            x = np.random.rand(*self.source.dom.shape) if self.source.dom.shape else np.random.rand()
            # For p_prime, we assume a dummy numeric sample; replace with real sampling.
            p_prime = np.random.rand(1)  # Placeholder sample.
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
