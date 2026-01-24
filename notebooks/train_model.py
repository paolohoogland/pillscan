import os
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms, models

from tqdm import tqdm # progress bar

from pill_dataset import PillDataset 

# matplotlib inline
import matplotlib.pyplot as plt

DATA_DIR = Path("../data.nosync/pills_raw/ogyeiv2")
TRAIN_DIR = DATA_DIR / "train" / "images"
VAL_DIR = DATA_DIR / "val" / "images"
TEST_DIR = DATA_DIR / "test" / "images"

BATCH_SIZE = 128  # optimal for RTX 4070 Super with this dataset
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
NUM_CLASSES = 112 # dataset specific

IMG_SIZE = 224 # resnet standard input size

# mps if available, else cpu
# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # for mac M2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # for nvidia gpus
print(f"Using device: {DEVICE}")

# for training, we need to resize, augment, normalize
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# for validation and testing, only resize and normalize (no augmentation)
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = PillDataset(TRAIN_DIR, transform=train_transforms)
val_dataset = PillDataset(VAL_DIR, transform=val_transforms)
test_dataset = PillDataset(TEST_DIR, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10, pin_memory=True, prefetch_factor=4, persistent_workers=True) # workers and pin_memory for performance
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10, pin_memory=True, prefetch_factor=4, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10, pin_memory=True, prefetch_factor=4, persistent_workers=True)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# USING RESNET50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# freeze all layers (only train the final classifier (fc))
for param in model.parameters():
    param.requires_grad = False

# replace the final fully connected layer for our 112 classes
# Linear(2048, 1000) -> Linear(2048, 112)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

print(f"Model: ResNet50 with {NUM_CLASSES} output classes")

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)  # fc layer

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0 # loss accumulator
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training"): # progress bar
        images, labels = images.to(device), labels.to(device)

        # zero gradients for each batch
        optimizer.zero_grad()
        # predict
        outputs = model(images)
        # compute loss
        loss = criterion(outputs, labels)
        # backpropagate
        loss.backward()
        # update weights
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1) # discard max values, keep indices (predicted classes)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item() # count correct predictions

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): 
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)

            # forward pass
            outputs = model(images) 
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

best_val_acc = 0.0
train_loss_values = []
val_loss_values = []
train_acc_values = []
val_acc_values = []

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    train_loss_values.append(train_loss)
    val_loss_values.append(val_loss)
    train_acc_values.append(train_acc)
    val_acc_values.append(val_acc)

    # save the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved.")
print("Training complete.")

model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, NUM_EPOCHS + 1), train_loss_values, label='Train Loss')
plt.plot(range(1, NUM_EPOCHS + 1), val_loss_values, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, NUM_EPOCHS + 1), train_acc_values, label='Train Acc')
plt.plot(range(1, NUM_EPOCHS + 1), val_acc_values, label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()