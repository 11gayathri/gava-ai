"""
train_waste_classifier.py - Train MobileNetV2 on real TrashNet dataset.
Protected for Windows multiprocessing and Intel OpenMP runtime.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "datasets" / "trashnet"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if (DATA_DIR / "data").exists():
        data_path = DATA_DIR / "data"
    else:
        data_path = DATA_DIR

    print(f"Loading real dataset from: {data_path}")

    # 1. Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Data loading
    full_dataset = datasets.ImageFolder(str(data_path), transform=train_transform)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_set, val_set = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    val_set.dataset.transform = val_transform

    # Set num_workers=0 to prevent Windows process spawn errors
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)

    print(f"Dataset summary: {train_size} training images, {val_size} validation images across {len(full_dataset.classes)} classes.")

    # 3. Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, 6)
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

    # 4. Training loop
    num_epochs = 10
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss/total:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            save_path = MODELS_DIR / "mobilenet_trashnet.pth"
            torch.save(model.state_dict(), str(save_path))

    print(f"\nTraining Complete. Best Validation Accuracy: {best_accuracy*100:.2f}%")
    print(f"Model saved to: {MODELS_DIR / 'mobilenet_trashnet.pth'}")


if __name__ == "__main__":
    main()