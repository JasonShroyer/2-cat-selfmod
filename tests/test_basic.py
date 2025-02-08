# tests/test_basic.py

import numpy as np
from basecat.objects import CatObject, TupleParamSpace
from basecat.morphisms import ParametricMorphism
from basecat.reparam2morph import Reparam2Morphism

def dummy_layer(theta, x):
    # A dummy function: assume theta is a scalar, x is a numpy array
    return theta * x

def test_forward():
    # Create domain and codomain objects
    X = CatObject("InputSpace", shape=(3,))
    Y = CatObject("OutputSpace", shape=(3,))
    # Create a simple parameter space with a single scalar
    param_space = TupleParamSpace((1.5,))
    # Create a ParametricMorphism using dummy_layer
    morph = ParametricMorphism(dom=X, cod=Y, param_obj=param_space, apply_fn=dummy_layer, name="DummyLayer")
    
    x = np.array([1.0, 2.0, 3.0])
    output = morph.forward((1.5,), x)  # passing the parameter as a tuple
    expected = 1.5 * x
    assert np.allclose(output, expected), "Forward computation did not match expected output."

def test_reparameterization():
    # Create two identical dummy layers.
    X = CatObject("InputSpace", shape=(3,))
    Y = CatObject("OutputSpace", shape=(3,))
    param_space = TupleParamSpace((2.0,))
    morph1 = ParametricMorphism(dom=X, cod=Y, param_obj=param_space, apply_fn=dummy_layer, name="Layer1")
    morph2 = ParametricMorphism(dom=X, cod=Y, param_obj=param_space, apply_fn=dummy_layer, name="Layer2")
    
    # Define a reparameterization: identity function
    reparam = lambda p: p  # identity reparameterization
    rho = Reparam2Morphism(source=morph1, target=morph2, reparam_map=reparam, name="IdentityReparam")
    
    # Check that the reparameterization holds
    assert rho.check_commute(test_samples=3), "Reparameterization did not commute as expected."

if __name__ == "__main__":
    test_forward()
    test_reparameterization()
    print("All basic tests passed.")
# tests/test_basic.py

import pytest
from basecat.morphisms import split_params, merge_params

def test_split_merge_params_success():
    # Create a composite parameter from two sample values.
    p1 = 3.14
    p2 = [1, 2, 3]
    composite = merge_params(p1, p2)
    # Check that split_params returns the original values.
    sp1, sp2 = split_params(composite)
    assert sp1 == p1, "First parameter not recovered correctly."
    assert sp2 == p2, "Second parameter not recovered correctly."

def test_split_params_failure():
    # Test that an incorrect parameter format raises a ValueError.
    with pytest.raises(ValueError):
        split_params(3.14)  # Not a tuple/list.
    with pytest.raises(ValueError):
        split_params((1, 2, 3))  # Tuple length not equal to 2.
