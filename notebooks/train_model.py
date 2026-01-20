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

DATA_DIR = Path("../data.nosync/pills_raw/ogyeiv2")
TRAIN_DIR = DATA_DIR / "train" / "images"
VAL_DIR = DATA_DIR / "val" / "images"
TEST_DIR = DATA_DIR / "test" / "images"

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 112 # dataset specific

IMG_SIZE = 224 # resnet standard input size

# mps if available, else cpu
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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