# src/self_modifying_mnist_training.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import copy

# Import our custom MNISTClassifier and self-modification functions.
from project_mnist import MNISTClassifier  # Your MNIST model
from self_modification import self_modify_fc1

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, targets) in enumerate(dataloader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define data transformations and load MNIST.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                               download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                              download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Instantiate the original model.
    model = MNISTClassifier().to(device)
    print("Original Model:")
    print(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for one epoch.
    print("Training original model for 1 epoch...")
    loss_before = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Average Loss (before self-modification): {loss_before:.4f}")
    
    # Evaluate the model before self-modification.
    acc_before = evaluate(model, test_loader, device)
    print(f"Test Accuracy (before self-modification): {acc_before:.2f}%")
    
    # Get a sample output from the original model.
    x_sample = torch.randn(1, 1, 28, 28).to(device)
    with torch.no_grad():
        sample_output_orig = model(x_sample)
    
    # Apply self-modification: reparameterize fc1.
    # For demonstration, we choose a scaling factor of 2.0.
    modified_model, reparam_map = self_modify_fc1(model, scale=2.0)
    modified_model = modified_model.to(device)
    print("Applied self-modification to fc1.")
    
    # Immediately compare sample outputs before any further training.
    with torch.no_grad():
        sample_output_mod = modified_model(x_sample)
    
    if torch.allclose(sample_output_orig, sample_output_mod, atol=1e-6):
        print("Immediate self-modification preserved sample output!")
    else:
        print("Immediate self-modification FAILED: sample outputs differ.")
        print("Original sample output:", sample_output_orig)
        print("Modified sample output:", sample_output_mod)
    
    # Now, continue training the modified model for one more epoch.
    optimizer = optim.Adam(modified_model.parameters(), lr=0.001)
    print("Training modified model for 1 epoch...")
    loss_after = train_epoch(modified_model, train_loader, criterion, optimizer, device)
    print(f"Average Loss (after self-modification training): {loss_after:.4f}")
    
    acc_after = evaluate(modified_model, test_loader, device)
    print(f"Test Accuracy (after self-modification training): {acc_after:.2f}%")

if __name__ == "__main__":
    main()
