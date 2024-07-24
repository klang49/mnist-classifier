import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Custom dataset class to handle image transformations
class MNISTDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.fromarray(np.array(item['image']))
        if self.transform:
            image = self.transform(image)
        return {'pixel_values': image, 'label': item['label']}

def train_model():
    print("Training the model...")
    # Load MNIST dataset
    mnist_dataset = load_dataset("mnist")

    # Define transforms
    transform = ToTensor()

    # Prepare datasets with the custom dataset class
    train_dataset = MNISTDataset(mnist_dataset["train"], transform=transform)
    test_dataset = MNISTDataset(mnist_dataset["test"], transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            images, labels = batch["pixel_values"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch["pixel_values"].to(device), batch["label"].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Accuracy: {100 * correct / total:.2f}%")

    print("Training finished!")

    # Save the model parameters
    torch.save(model.state_dict(), 'bin/model_weights.pth')

def analyze_model():
    print("Analyzing the model...")
    # Load the model
    model = Net()
    model.load_state_dict(torch.load('bin/model_weights.pth'))
    model.eval()

    # Load test dataset
    mnist_dataset = load_dataset("mnist")
    test_dataset = MNISTDataset(mnist_dataset["test"], transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Collect misclassified samples
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    misclassified_images = []
    misclassified_labels = []
    misclassified_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch["pixel_values"].to(device), batch["label"].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Find misclassified samples
            misclassified_mask = (predicted != labels)
            misclassified_images.extend(images[misclassified_mask].cpu())
            misclassified_labels.extend(labels[misclassified_mask].cpu())
            misclassified_predictions.extend(predicted[misclassified_mask].cpu())
            
            if len(misclassified_images) >= 10:
                break

    # Function to show images
    def imshow(img,ax):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)),cmap='gray')

    # Plot 10 misclassified samples
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < len(misclassified_images):
            imshow(misclassified_images[i],ax)
            ax.set_title(f'True: {misclassified_labels[i]}\nPred: {misclassified_predictions[i]}')
            ax.axis('off')

    plt.tight_layout()
    os.makedirs(f'bin/misclassified_samples/', exist_ok=True)
    plt.savefig(f'bin/misclassified_samples/sample_{i+1}.png')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='MNIST Model Training and Analysis')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--analysis', action='store_true', help='Analyze the model')
    args = parser.parse_args()

    if args.train:
        train_model()
    if args.analysis:
        analyze_model()
    if not (args.train or args.analysis):
        print("Please specify either --train or --analysis")

if __name__ == "__main__":
    main()
