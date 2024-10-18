from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from network import Net
import os
import shutil
import argparse

def train(
        model,
        train_loader, criterion, optimizer,
        device,
        epochs=1,
        start_epoch=0, start_loss=0.0,
        log_rate=.05,
        checkpoints=True, checkpoint_interval=1,
        checkpoint_dir="checkpoints", clear_checkpoint_dir=False, ignore_non_empty_checkpoint_dir=False
    ):

    if checkpoints:
        if not os.path.exists(checkpoint_dir):
            # create checkpoint dir when it does not already exist
            os.makedirs(checkpoint_dir)
            print(f"Created checkpoint directory \"{checkpoint_dir}\"")
        elif not ignore_non_empty_checkpoint_dir:
            if not clear_checkpoint_dir and len(os.listdir(checkpoint_dir)) > 0:
                raise Exception("Checkpoint directory is not empty. Either rename old checkpoint dir or use --remove_old_checkpoints")
            
            # clear checkpoint dir
            for filename in os.listdir(checkpoint_dir):
                file_path = os.path.join(checkpoint_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

    model.train().to(device)

    losses_per_epoch = []

    for epoch in range(start_epoch, epochs):
        total_loss = start_loss  # Accumulate total loss over the epoch
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

        # Save checkpoint at the end of the epoch if required
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1, avg_loss_epoch, checkpoint_dir)

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

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    # Create a dynamic filename with the epoch and loss included
    file_name = f"{checkpoint_dir}/checkpoint_epoch_{epoch}_loss_{loss:.4f}.pth"
    
    # Save the model, optimizer, and metadata
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, file_name)
    print(f"Saved checkpoint {file_name}")

def latest_checkpoint(checkpoint_dir="checkpoints"):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        raise Exception("No checkpoints found in the directory")

    # Extract epoch numbers from filenames and find the one with the highest epoch
    latest_checkpoint = max(checkpoints, key=lambda f: int(f.split('_')[2]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

    return checkpoint_path

def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: Resuming training from epoch {epoch}")
    return epoch, loss

def save_model(model, file_path="trained_model.pth"):
    torch.save(model.state_dict(), file_path)
    print(f"Trained model saved to {file_path}")

if torch.cuda.is_available():
    print("Found CUDA device")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# args
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--remove_old_checkpoints', action='store_true', help='Remove old checkpoints if present')
parser.add_argument('-c', '--start_from_checkpoint', action='store_true', help='Start from the latest checkpoint. Does nothing when no checkpoint is present.')
args = parser.parse_args()

# remove_old_checkpoints not allowed when starting from checkpoint
if args.remove_old_checkpoints and args.start_from_checkpoint:
    raise Exception("Cannot remove old checkpoints when starting from checkpoint")

# Define a simple transform (convert image to tensor and normalize)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Download and load the training and test sets
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.9

# Load the datasets into simple data loaders (no batching for now)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define the model, loss function, and optimizer
model = Net().to(device)
criterion = nn.NLLLoss()  # Negative Log-Likelihood Loss
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

start_epoch = 0
start_loss = 0.0

if args.start_from_checkpoint:
    checkpoint_path = latest_checkpoint()
    start_epoch, start_loss = load_checkpoint(model, optimizer, checkpoint_path, device)

losses = train(
            model, train_loader, criterion, optimizer, epochs=EPOCHS, device=device,
            start_epoch=start_epoch, start_loss=start_loss,
            clear_checkpoint_dir=args.remove_old_checkpoints,
            ignore_non_empty_checkpoint_dir=args.start_from_checkpoint
        )
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