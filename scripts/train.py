from pathlib import Path
import random
import json

import numpy as np
import rasterio
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("data/labeled")

TRAIN_IMG_DIR = ROOT / "Train_Dataset" / "Train_TIF"
TRAIN_MASK_DIR = ROOT / "Train_Dataset" / "Mask_PNG"

TEST_IMG_DIR = ROOT / "Test_Dataset" / "Test_TIF"
TEST_MASK_DIR = ROOT / "Test_Dataset" / "Mask_PNG"

OUTPUT_DIR = Path("reports/training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 1e-3

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 70)
print("ATHENA CV — ICEBERG SEGMENTATION TRAINING")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Image size: {IMAGE_SIZE} × {IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print()


# ============================================================
# DATASET PAIRING
# ============================================================

def build_pairs(image_dir, mask_dir):
    images = sorted(image_dir.glob("*.tif"))
    masks = sorted(mask_dir.glob("*.png"))

    mask_map = {
        p.stem.replace("_mask", ""): p
        for p in masks
    }

    pairs = [
        (img, mask_map[img.stem])
        for img in images
        if img.stem in mask_map
    ]

    return pairs


train_pairs = build_pairs(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
test_pairs = build_pairs(TEST_IMG_DIR, TEST_MASK_DIR)

print(f"Training pairs: {len(train_pairs)}")
print(f"Test pairs:     {len(test_pairs)}")

if len(train_pairs) == 0:
    raise RuntimeError("No training pairs found.")

if len(test_pairs) == 0:
    raise RuntimeError("No test pairs found.")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_sar(path):
    with rasterio.open(path) as src:
        image = src.read(1).astype(np.float32)

    # Replace invalid values
    valid = np.isfinite(image)

    if not valid.any():
        raise ValueError(f"No valid pixels in {path}")

    values = image[valid]

    # Robust per-image normalization.
    # This reduces the effect of extreme SAR values while
    # preserving the relative bright/dark structure.
    lo, hi = np.percentile(values, [1, 99])

    if hi <= lo:
        lo = values.min()
        hi = values.max()

    image = np.clip(image, lo, hi)
    image = (image - lo) / (hi - lo + 1e-8)

    image[~valid] = 0.0

    return image


def load_mask(path):
    mask = np.array(Image.open(path).convert("L"))

    # Binary mask:
    # 0   = background
    # >0  = iceberg
    mask = (mask > 127).astype(np.float32)

    return mask


def resize_image(image):
    image = Image.fromarray(image.astype(np.float32), mode="F")
    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    return np.array(image, dtype=np.float32)


def resize_mask(mask):
    mask = Image.fromarray(
        (mask * 255).astype(np.uint8),
        mode="L"
    )

    mask = mask.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.NEAREST
    )

    return np.array(mask, dtype=np.float32) / 255.0


# ============================================================
# DATA AUGMENTATION
# ============================================================

def augment(image, mask):
    if random.random() < 0.5:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

    if random.random() < 0.5:
        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()

    # 90-degree rotations preserve SAR structure without
    # interpolation artifacts.
    if random.random() < 0.5:
        k = random.randint(1, 3)
        image = np.rot90(image, k).copy()
        mask = np.rot90(mask, k).copy()

    return image, mask


# ============================================================
# PYTORCH DATASET
# ============================================================

class IcebergDataset(Dataset):

    def __init__(self, pairs, training=False):
        self.pairs = pairs
        self.training = training

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        image_path, mask_path = self.pairs[idx]

        image = load_sar(image_path)
        mask = load_mask(mask_path)

        image = resize_image(image)
        mask = resize_mask(mask)

        if self.training:
            image, mask = augment(image, mask)

        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask


# ============================================================
# U-NET
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(1, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(self.pool(e1))

        e3 = self.enc3(self.pool(e2))

        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.output(d1)


# ============================================================
# LOSS
# ============================================================

def dice_loss(logits, targets):

    probabilities = torch.sigmoid(logits)

    smooth = 1e-6

    intersection = (
        probabilities * targets
    ).sum(dim=(1, 2, 3))

    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )

    dice = (
        2 * intersection + smooth
    ) / (
        denominator + smooth
    )

    return 1 - dice.mean()


def combined_loss(logits, targets):

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets
    )

    dice = dice_loss(
        logits,
        targets
    )

    return bce + dice


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(logits, targets):

    probabilities = torch.sigmoid(logits)
    predictions = probabilities > 0.5
    targets = targets > 0.5

    tp = (predictions & targets).sum().item()
    fp = (predictions & ~targets).sum().item()
    fn = (~predictions & targets).sum().item()
    tn = (~predictions & ~targets).sum().item()

    eps = 1e-7

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    dice = (
        2 * tp
    ) / (
        2 * tp + fp + fn + eps
    )

    iou = (
        tp
    ) / (
        tp + fp + fn + eps
    )

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(model, loader, optimizer):

    model.train()

    total_loss = 0.0

    for batch_idx, (images, masks) in enumerate(loader):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)

        loss = combined_loss(
            logits,
            masks
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(model, loader):

    model.eval()

    total_loss = 0.0

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    for images, masks in loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        logits = model(images)

        loss = combined_loss(
            logits,
            masks
        )

        total_loss += loss.item()

        metrics = calculate_metrics(
            logits,
            masks
        )

        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]
        total_tn += metrics["tn"]

    eps = 1e-7

    precision = total_tp / (
        total_tp + total_fp + eps
    )

    recall = total_tp / (
        total_tp + total_fn + eps
    )

    dice = (
        2 * total_tp
    ) / (
        2 * total_tp + total_fp + total_fn + eps
    )

    iou = (
        total_tp
    ) / (
        total_tp + total_fp + total_fn + eps
    )

    return {
        "loss": total_loss / len(loader),
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


# ============================================================
# SAVE VISUAL PREDICTIONS
# ============================================================

@torch.no_grad()
def save_predictions(model, dataset, epoch):

    model.eval()

    prediction_dir = (
        OUTPUT_DIR /
        "predictions" /
        f"epoch_{epoch:03d}"
    )

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save first 5 examples
    for i in range(min(5, len(dataset))):

        image, mask = dataset[i]

        logits = model(
            image.unsqueeze(0).to(DEVICE)
        )

        probability = torch.sigmoid(
            logits
        )[0, 0].cpu().numpy()

        prediction = (
            probability > 0.5
        ).astype(np.uint8) * 255

        Image.fromarray(
            prediction
        ).save(
            prediction_dir /
            f"{i:02d}_prediction.png"
        )

        Image.fromarray(
            (mask[0].numpy() * 255).astype(np.uint8)
        ).save(
            prediction_dir /
            f"{i:02d}_ground_truth.png"
        )

        Image.fromarray(
            (image[0].numpy() * 255).astype(np.uint8)
        ).save(
            prediction_dir /
            f"{i:02d}_sar.png"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    train_dataset = IcebergDataset(
        train_pairs,
        training=True
    )

    test_dataset = IcebergDataset(
        test_pairs,
        training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = UNet().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    history = []

    best_dice = -1.0

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer
        )

        metrics = evaluate(
            model,
            test_loader
        )

        scheduler.step(
            metrics["dice"]
        )

        current_lr = optimizer.param_groups[0]["lr"]

        result = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
            "learning_rate": current_lr,
        }

        history.append(result)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"test_loss={metrics['loss']:.4f} | "
            f"dice={metrics['dice']:.4f} | "
            f"iou={metrics['iou']:.4f} | "
            f"precision={metrics['precision']:.4f} | "
            f"recall={metrics['recall']:.4f}"
        )

        if metrics["dice"] > best_dice:

            best_dice = metrics["dice"]

            torch.save(
                model.state_dict(),
                OUTPUT_DIR / "best_unet.pth"
            )

            print(
                f"  ✓ New best model "
                f"(Dice={best_dice:.4f})"
            )

        # Save prediction snapshots
        if epoch == 1 or epoch % 5 == 0:
            save_predictions(
                model,
                test_dataset,
                epoch
            )

    with open(
        OUTPUT_DIR / "training_history.json",
        "w"
    ) as f:
        json.dump(
            history,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Test Dice: {best_dice:.4f}")
    print()
    print("Model:")
    print(OUTPUT_DIR / "best_unet.pth")
    print()
    print("History:")
    print(OUTPUT_DIR / "training_history.json")
    print()
    print("Predictions:")
    print(OUTPUT_DIR / "predictions")


if __name__ == "__main__":
    main()
