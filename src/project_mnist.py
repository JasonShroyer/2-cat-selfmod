# src/project_mnist.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Import our custom layers from our library.
from basecat.layers import LinearLayer, ReLUActivation
from basecat.objects import CatObject, TupleParamSpace  # (optional, for documentation)

# Define a simple MNIST classifier using our custom layers.
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        # MNIST images are 28x28; we'll flatten them to a vector of size 784.
        self.flatten = nn.Flatten()
        
        # We define our network as follows:
        # 784 -> 128 -> ReLU -> 10
        self.fc1 = LinearLayer(in_features=784, out_features=128)
        self.relu = ReLUActivation()
        self.fc2 = LinearLayer(in_features=128, out_features=10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input images.
        x = self.flatten(x)
        # Pass through the first linear layer.
        x = self.fc1(x)
        # Apply ReLU activation.
        x = self.relu(x)
        # Pass through the second linear layer.
        x = self.fc2(x)
        return x

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transformations for MNIST: convert images to tensor and normalize.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Download and load MNIST training and test datasets.
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                               download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                              download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Instantiate our MNIST classifier.
    model = MNISTClassifier().to(device)
    print(model)

    # Define loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop (for demonstration, we'll train for 2 epochs).
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Zero the gradients.
            optimizer.zero_grad()
            # Forward pass.
            outputs = model(data)
            loss = criterion(outputs, targets)
            # Backward pass.
            loss.backward()
            # Update parameters.
            optimizer.step()
            
            running_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

    # Evaluate on the test set.
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            # Get predictions from the maximum value.
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
