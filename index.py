from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from network import Net

def train(model, train_loader, criterion, optimizer, device, epochs=1, log_rate=.05):
    model.train().to(device)

    losses_per_epoch = []

    for epoch in range(epochs):
        total_loss = 0.0  # Accumulate total loss over the epoch
        total_batches = len(train_loader)  # Total number of batches

        log_interval = int(total_batches * log_rate)

        progress_bar = tqdm(enumerate(train_loader), total=total_batches, desc=f"Epoch {epoch+1}/{epochs}")

        avg_loss = 0
        for i, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)

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

            # Accumulate total loss
            total_loss += loss.item()

            progress_bar.set_postfix({"Avg. Loss": f"{avg_loss:.4f}", "Loss": f"{loss.item():.4f}"})

            if (i+1) % log_interval == 0:
                avg_loss = total_loss/(i+1)

        # Calculate average loss for the entire epoch
        avg_loss_epoch = total_loss / total_batches
        losses_per_epoch.append(avg_loss_epoch)

        # Print average loss for the whole epoch
        print(f"Epoch [{epoch+1}/{epochs}] - Average Loss: {avg_loss_epoch:.4f}")

    return losses_per_epoch

def test(model, test_loader, device):
    model.eval().to(device)
    correct = 0
    total = 0
    with torch.no_grad():  # No need to compute gradients during testing
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

def save_model(model, file_path="trained_model.pth"):
    torch.save(model.state_dict(), file_path)
    print(f"Trained model saved to {file_path}")

if torch.cuda.is_available():
    print("Found CUDA device")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple transform (convert image to tensor and normalize)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Download and load the training and test sets
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.9

# Load the datasets into simple data loaders (no batching for now)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define the model, loss function, and optimizer
model = Net()
criterion = nn.NLLLoss()  # Negative Log-Likelihood Loss
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

losses = train(model, train_loader, criterion, optimizer, epochs=EPOCHS, device=device)
test(model, test_loader, device)

save_model(model)

# Plot the loss curve
plt.figure(figsize=(10, EPOCHS))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()