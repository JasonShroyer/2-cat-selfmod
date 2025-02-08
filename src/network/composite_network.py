import torch
import torch.nn as nn
from basecat.objects import CatObject
from basecat.diff_morphisms import DifferentiableMorphism

class CompositeNetwork(nn.Module):
    """
    An example composite network that chains two differentiable morphisms.
    Each morphism performs a simple linear scaling.
    """
    def __init__(self):
        super(CompositeNetwork, self).__init__()
        # Define domain and codomain objects.
        self.input_space = CatObject("InputSpace", shape=(1,))
        self.hidden_space = CatObject("HiddenSpace", shape=(1,))
        self.output_space = CatObject("OutputSpace", shape=(1,))

        # Create two differentiable morphisms (layers) with linear scaling.
        self.layer1 = DifferentiableMorphism(
            dom=self.input_space,
            cod=self.hidden_space,
            apply_fn=lambda theta, x: theta * x,  # Simple scaling: y = theta * x
            init_param=torch.tensor([2.0]),
            name="Layer1"
        )
        self.layer2 = DifferentiableMorphism(
            dom=self.hidden_space,
            cod=self.output_space,
            apply_fn=lambda theta, x: theta * x,  # Simple scaling: y = theta * x
            init_param=torch.tensor([3.0]),
            name="Layer2"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Chain the two layers: first scale by layer1, then scale by layer2.
        h = self.layer1(x)
        y = self.layer2(h)
        return y

if __name__ == "__main__":
    # Quick test of the composite network.
    net = CompositeNetwork()
    x = torch.tensor([1.0], requires_grad=True)
    y = net(x)
    print(f"Input: {x.item()}, Output: {y.item()}")
    
    # Compute a dummy loss and perform a backward pass.
    loss = y.sum()
    loss.backward()
    print("Layer1 gradient:", net.layer1.param.grad)
    print("Layer2 gradient:", net.layer2.param.grad)
