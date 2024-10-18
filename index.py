import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms

from network import Net

def train(model, train_loader, criterion, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
        for images, labels in train_loader:

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(images)

            # Calculate loss
            loss = criterion(output, labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # No need to compute gradients during testing
        for images, labels in test_loader:
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Define a simple transform (convert image to tensor and normalize)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Download and load the training and test sets
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Load the datasets into simple data loaders (no batching for now)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define the model, loss function, and optimizer
model = Net()
criterion = nn.NLLLoss()  # Negative Log-Likelihood Loss
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

train(model, train_loader, criterion, optimizer, epochs=5)  # Train for 5 epochs
test(model, test_loader)