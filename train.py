"""
Diabetic Retinopathy Detection — Training Script
=================================================

Transfer learning (EfficientNet-B0) on the APTOS 2019 Blindness Detection
dataset (Kaggle). Classifies retinal fundus images into 5 severity grades:

    0 - No DR
    1 - Mild
    2 - Moderate
    3 - Severe
    4 - Proliferative DR

WHERE TO RUN THIS
------------------
Run this in Google Colab or Kaggle Notebooks (both give a free GPU).
This sandbox has no GPU and no access to Kaggle, so this script is meant
to be copied there, not run here.

DATASET
-------
1. Go to: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
   (free Kaggle account required)
2. Download and unzip so you have:
       train.csv              (columns: id_code, diagnosis)
       train_images/           (folder of .png fundus photos)
3. In Colab, either upload the zip or use the Kaggle API:
       !pip install kaggle
       !kaggle competitions download -c aptos2019-blindness-detection
       !unzip aptos2019-blindness-detection.zip -d data/

OUTPUT
------
best_model.pth   -> trained weights, copy this into app/ for the Streamlit app
training_log.csv -> per-epoch loss/accuracy/kappa, useful for a Power BI chart
confusion_matrix.png
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = "data"                       # expects data/train.csv, data/train_images/
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LR = 3e-4
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------
class RetinopathyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row.id_code}.png")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row.diagnosis)
        return image, label


train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# MODEL — EfficientNet-B0 backbone, fine-tuned head
# ---------------------------------------------------------------------------
def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model.to(DEVICE)


# ---------------------------------------------------------------------------
# TRAIN / EVAL LOOPS
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    return avg_loss, acc, kappa, all_preds, all_labels


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    train_df, val_df = train_test_split(
        df, test_size=0.15, stratify=df["diagnosis"], random_state=42
    )

    train_ds = RetinopathyDataset(train_df, os.path.join(DATA_DIR, "train_images"), train_transforms)
    val_ds = RetinopathyDataset(val_df, os.path.join(DATA_DIR, "train_images"), val_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model()

    # class-weighted loss since APTOS is imbalanced (most images are "No DR")
    class_counts = train_df["diagnosis"].value_counts().sort_index().values
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
    class_weights = (class_weights / class_weights.sum() * NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    best_kappa = -1
    log_rows = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc, train_kappa, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_kappa, val_preds, val_labels = run_epoch(model, val_loader, criterion)
        scheduler.step(val_kappa)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train_loss {train_loss:.4f} acc {train_acc:.3f} kappa {train_kappa:.3f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.3f} kappa {val_kappa:.3f} | {elapsed:.0f}s")

        log_rows.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                              train_kappa=train_kappa, val_loss=val_loss, val_acc=val_acc,
                              val_kappa=val_kappa))

        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  -> saved new best model (kappa={best_kappa:.3f})")

    pd.DataFrame(log_rows).to_csv("training_log.csv", index=False)

    # final confusion matrix on the last validation pass
    cm = confusion_matrix(val_labels, val_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=150)

    print(f"\nDone. Best validation quadratic-weighted kappa: {best_kappa:.3f}")
    print("Copy best_model.pth into app/ to use it in the Streamlit app.")


if __name__ == "__main__":
    main()
