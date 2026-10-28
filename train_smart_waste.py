"""
train_smart_waste.py
====================
Trains MobileNetV3 on 4 fine-grained classes:
  - food_waste
  - plastic_bag
  - plastic_rigid
  - paper_cardboard

Saves model to: models/smart_waste_mobilenet.pth
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "datasets" / "smart_waste_dataset"
SAVE_PATH = ROOT / "models" / "smart_waste_mobilenet.pth"
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using compute device: {device}")

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(str(DATA_DIR), transform=train_transforms)
    class_names = dataset.classes
    print(f"[Training] Classes ({len(class_names)}): {class_names}")

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    # Freeze base feature extractor
    for p in model.features.parameters():
        p.requires_grad = False

    in_feat = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_feat, len(class_names))
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.0015)

    epochs = 6
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

        acc = (correct / total) * 100.0
        print(f"  Epoch [{epoch+1}/{epochs}] Loss: {running_loss/total:.4f} | Accuracy: {acc:.2f}%")

    torch.save({
        "state_dict": model.state_dict(),
        "classes": class_names
    }, SAVE_PATH)
    print(f"\n[Training Complete] Saved to: {SAVE_PATH}")


if __name__ == "__main__":
    train()