# src/composition/compose_2morph.py

from basecat.reparam2morph import Reparam2Morphism
from basecat.morphisms import ParametricMorphism

def vertical_compose(rho1: Reparam2Morphism, rho2: Reparam2Morphism) -> Reparam2Morphism:
    """
    Vertically compose two 2-morphisms:
      If rho1: (P, f) => (P', f') and rho2: (P', f') => (P'', f''),
      then the vertical composition rho: (P, f) => (P'', f'')
      has a reparameterization map defined as:
           r = rho1.reparam_map o rho2.reparam_map
    """
    if rho1.target != rho2.source:
        raise ValueError("Cannot compose vertically: target of first 2-morphism does not match source of second.")

    def composed_reparam(p_double_prime):
        return rho1.reparam_map(rho2.reparam_map(p_double_prime))
    
    new_name = f"({rho2.name} ∘ {rho1.name})"
    return Reparam2Morphism(source=rho1.source, target=rho2.target, reparam_map=composed_reparam, name=new_name)

def horizontal_compose(rho_left: Reparam2Morphism, rho_right: Reparam2Morphism) -> Reparam2Morphism:
    """
    Horizontally compose two 2-morphisms.
    Suppose:
      - rho_left: (P, f) => (P', f')
      - rho_right: (Q, g) => (Q', g')
    Then the horizontal composition yields a 2-morphism for the composite morphisms:
         (Q ⊗ P, g ∘ f) => (Q' ⊗ P', g' ∘ f')
    with the reparameterization map defined as:
         (q', p') -> (rho_right.reparam_map(q'), rho_left.reparam_map(p'))
    """
    # Compose the corresponding 1-morphisms.
    source_composite = rho_left.source.compose(rho_right.source)
    target_composite = rho_left.target.compose(rho_right.target)

    def composed_reparam(param_tuple):
        # Expect param_tuple to be a tuple (p_prime, q_prime)
        p_prime, q_prime = param_tuple
        return (rho_left.reparam_map(p_prime), rho_right.reparam_map(q_prime))
    
    new_name = f"({rho_right.name} ∗ {rho_left.name})"
    return Reparam2Morphism(source=source_composite, target=target_composite, reparam_map=composed_reparam, name=new_name)
