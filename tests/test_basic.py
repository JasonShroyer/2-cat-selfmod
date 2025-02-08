import numpy as np
import pytest
from basecat.objects import CatObject, TupleParamSpace
from basecat.morphisms import ParametricMorphism, split_params, merge_params
from basecat.reparam2morph import Reparam2Morphism

# Dummy function definitions
def dummy_layer_scaling(theta, x):
    # Scaling function: returns theta * x.
    return theta * x

def dummy_layer_identity(theta, x):
    # Identity function: returns x regardless of theta.
    return x

def dummy_layer_scaling_source(theta, x):
    # Source scaling function: returns theta * x.
    return theta * x

def dummy_layer_scaling_target(theta, x):
    # Target scaling function: returns 2 * theta * x.
    return 2 * theta * x

# Test: forward computation using a scaling function
def test_forward():
    # Create domain and codomain objects.
    X = CatObject("InputSpace", shape=(3,))
    Y = CatObject("OutputSpace", shape=(3,))
    # Create a simple parameter space with a single scalar.
    param_space = TupleParamSpace((1.5,))
    # Create a ParametricMorphism using dummy_layer_scaling.
    morph = ParametricMorphism(
        dom=X, 
        cod=Y, 
        param_obj=param_space, 
        apply_fn=dummy_layer_scaling, 
        name="DummyLayer"
    )
    
    x = np.array([1.0, 2.0, 3.0])
    output = morph.forward((1.5,), x)  # passing the parameter as a tuple.
    expected = 1.5 * x
    assert np.allclose(output, expected), "Forward computation did not match expected output."

# Test: parameter splitting/merging
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

# Test: identity reparameterization
def test_identity_reparam():
    # Setup dummy objects.
    X = CatObject("InputSpace", shape=(1,))
    Y = CatObject("OutputSpace", shape=(1,))
    param_space = TupleParamSpace((1.0,))  # simple scalar parameter wrapped as a tuple.
    
    morph_source = ParametricMorphism(
        X, Y, param_space, dummy_layer_identity, name="Source"
    )
    morph_target = ParametricMorphism(
        X, Y, param_space, dummy_layer_identity, name="Target"
    )
    
    # Identity reparam: r(p') = p'
    identity_reparam = lambda p: p
    rho = Reparam2Morphism(morph_source, morph_target, identity_reparam, name="IdentityReparam")
    
    assert rho.check_commute(test_samples=5, tol=1e-6, use_torch=False), "Identity reparameterization failed."

# Test: scaling reparameterization
def test_scaling_reparam():
    # Setup dummy objects.
    X = CatObject("InputSpace", shape=(1,))
    Y = CatObject("OutputSpace", shape=(1,))
    param_space = TupleParamSpace((1.0,))
    
    # Source uses dummy_layer_scaling_source: returns theta * x.
    morph_source = ParametricMorphism(
        X, Y, param_space, dummy_layer_scaling_source, name="SourceScaling"
    )
    # Target uses dummy_layer_scaling_target: returns 2 * theta * x.
    morph_target = ParametricMorphism(
        X, Y, param_space, dummy_layer_scaling_target, name="TargetScaling"
    )
    
    # Define a reparameterization that scales p' by 2 so that:
    # target.forward(p', x) = 2 * p' * x,
    # source.forward(r(p'), x) = dummy_layer_scaling_source(2 * p', x) = (2 * p') * x.
    scaling_reparam = lambda p: 2 * p
    rho = Reparam2Morphism(morph_source, morph_target, scaling_reparam, name="ScalingReparam")
    
    assert rho.check_commute(test_samples=5, tol=1e-6, use_torch=False), "Scaling reparameterization failed."

if __name__ == "__main__":
    test_forward()
    test_split_merge_params_success()
    test_split_params_failure()
    test_identity_reparam()
    test_scaling_reparam()
    print("All basic tests passed.")

def test_identity_reparam_torch():
    X = CatObject("InputSpace", shape=(1,))
    Y = CatObject("OutputSpace", shape=(1,))
    param_space = TupleParamSpace((1.0,))
    
    morph_source = ParametricMorphism(X, Y, param_space, dummy_layer_identity, name="Source")
    morph_target = ParametricMorphism(X, Y, param_space, dummy_layer_identity, name="Target")
    
    identity_reparam = lambda p: p
    rho = Reparam2Morphism(morph_source, morph_target, identity_reparam, name="IdentityReparamTorch")
    
    assert rho.check_commute(test_samples=5, tol=1e-6, use_torch=True), "Torch-based identity reparameterization failed."
