import os
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt


# =====================================================
# DEVICE
# =====================================================

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", DEVICE)


# =====================================================
# PATHS
# =====================================================

RAW_DATA_DIR = "/Users/nehabalamurugan/Desktop/Dr. V. BALAMURUGAN_MAT IMAGES"

DATA_DIR = "data"

CLASS_NAMES = [
    "Below 20%",
    "20-40%",
    "40-50%",
    "50-60%",
    "60-70%",
    "More than 70%"
]

CFG = {
    "batch_size": 16,
    "epochs": 40,
    "lr": 0.0003
}

IMG_EXTS = {".jpg", ".jpeg", ".png"}


# =====================================================
# TRANSFORMS
# =====================================================

train_transform = transforms.Compose([

    transforms.Resize((256,256)),

    transforms.RandomCrop(224),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(25),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =====================================================
# PREPARE DATASET
# =====================================================

def prepare_dataset():

    if os.path.exists(DATA_DIR):
        print("Dataset already prepared")
        return

    for split in ["train", "val"]:

        for cls in CLASS_NAMES:

            os.makedirs(
                f"{DATA_DIR}/{split}/{cls}",
                exist_ok=True
            )

    for cls in CLASS_NAMES:

        folder = os.path.join(RAW_DATA_DIR, cls)

        images = [
            x for x in os.listdir(folder)
            if Path(x).suffix.lower() in IMG_EXTS
        ]

        random.shuffle(images)

        split_index = int(0.8 * len(images))

        train_imgs = images[:split_index]

        val_imgs = images[split_index:]

        for img in train_imgs:

            shutil.copy(
                os.path.join(folder, img),
                f"{DATA_DIR}/train/{cls}/{img}"
            )

        for img in val_imgs:

            shutil.copy(
                os.path.join(folder, img),
                f"{DATA_DIR}/val/{cls}/{img}"
            )

    print("Dataset prepared")


# =====================================================
# DATALOADERS
# =====================================================

def get_dataloaders():

    train_ds = datasets.ImageFolder(
        f"{DATA_DIR}/train",
        transform=train_transform
    )

    val_ds = datasets.ImageFolder(
        f"{DATA_DIR}/val",
        transform=val_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CFG["batch_size"]
    )

    idx_to_class = {
        v:k for k,v in train_ds.class_to_idx.items()
    }

    return train_loader, val_loader, idx_to_class


# =====================================================
# MODEL
# =====================================================

class MATClassifier(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.model = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )

        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(

            nn.Dropout(0.5),

            nn.Linear(in_features, 512),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):

        return self.model(x)


# =====================================================
# TRAIN
# =====================================================

def train():

    prepare_dataset()

    train_loader, val_loader, idx_to_class = get_dataloaders()

    model = MATClassifier(len(idx_to_class)).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5
    )

    best_acc = 0

    train_accs = []
    val_accs = []

    for epoch in range(CFG["epochs"]):

        # ================= TRAIN =================

        model.train()

        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            preds = outputs.argmax(1)

            train_correct += (preds == labels).sum().item()

            train_total += labels.size(0)

        train_acc = train_correct / train_total

        # ================= VALIDATION =================

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)

                labels = labels.to(DEVICE)

                outputs = model(images)

                preds = outputs.argmax(1)

                val_correct += (preds == labels).sum().item()

                val_total += labels.size(0)

        val_acc = val_correct / val_total

        train_accs.append(train_acc)

        val_accs.append(val_acc)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{CFG['epochs']} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # SAVE BEST MODEL

        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(
                model.state_dict(),
                "best_model.pth"
            )

    print("\nBest Validation Accuracy:", best_acc)

    # ================= PLOTS =================

    plt.plot(train_accs, label="Train")

    plt.plot(val_accs, label="Validation")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title("Training vs Validation Accuracy")

    plt.legend()

    plt.show()

    return model, idx_to_class


# =====================================================
# LOAD MODEL
# =====================================================

def load_model(idx_to_class):

    model = MATClassifier(
        len(idx_to_class)
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            "best_model.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    return model


# =====================================================
# PREDICT
# =====================================================

def predict(model, image_path, idx_to_class):

    image = Image.open(image_path).convert("RGB")

    tensor = val_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(tensor)

        probs = torch.softmax(output, dim=1)

        pred = probs.argmax(1).item()

    class_name = idx_to_class[pred]

    confidence = probs[0][pred].item()

    return class_name, confidence, image


# =====================================================
# SHOW RESULT
# =====================================================

def show_result(class_name, confidence, image):

    plt.figure(figsize=(5,5))

    plt.imshow(image)

    plt.title(
        f"{class_name} | {confidence:.2%}",
        fontsize=16
    )

    plt.axis("off")

    plt.show()