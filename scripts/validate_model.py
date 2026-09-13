from pathlib import Path
import sys

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import UNet, IcebergDataset, DEVICE


ROOT = Path("data/labeled/Validation_Dataset")
MODEL_PATH = Path("reports/training/best_unet.pth")


def build_validation_pairs():
    pairs = []

    # --------------------------------------------------------
    # Extended Validation
    # --------------------------------------------------------
    ext_img_dir = ROOT / "Extended_Validation" / "TIF"
    ext_mask_dir = ROOT / "Extended_Validation" / "Mask_PNG"

    for image_path in sorted(ext_img_dir.glob("*.tif")):
        mask_path = ext_mask_dir / f"{image_path.stem}_mask.png"

        if mask_path.exists():
            pairs.append((image_path, mask_path))

    # --------------------------------------------------------
    # Monthly Validation
    # Mask filename contains only the timestamp.
    # Extract timestamp from the TIFF filename.
    # --------------------------------------------------------
    monthly_img_dir = ROOT / "Monthly_Validation_Adelie" / "TIF"
    monthly_mask_dir = ROOT / "Monthly_Validation_Adelie" / "Mask_PNG"

    for image_path in sorted(monthly_img_dir.glob("*.tif")):

        parts = image_path.name.split("_")

        timestamp = None

        for part in parts:
            if "T" in part and len(part) >= 15:
                timestamp = part
                break

        if timestamp is None:
            print(f"⚠ Could not find timestamp: {image_path.name}")
            continue

        mask_path = monthly_mask_dir / f"{timestamp}_mask.png"

        if mask_path.exists():
            pairs.append((image_path, mask_path))
        else:
            print(f"⚠ Missing mask: {image_path.name}")

    return pairs


def calculate_metrics(prediction, target):

    prediction = prediction.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(prediction, target).sum()
    fp = np.logical_and(prediction, ~target).sum()
    fn = np.logical_and(~prediction, target).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)

    iou = tp / (tp + fp + fn + 1e-8)

    precision = tp / (tp + fp + 1e-8)

    recall = tp / (tp + fn + 1e-8)

    return dice, iou, precision, recall


def main():

    print("=" * 70)
    print("ATHENA CV — INDEPENDENT VALIDATION")
    print("=" * 70)

    pairs = build_validation_pairs()

    print(f"Validation pairs found: {len(pairs)}")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print()

    if len(pairs) != 20:
        print("⚠ WARNING: Expected 20 validation pairs.")

    # Load model
    model = UNet().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    dataset = IcebergDataset(
        pairs,
        training=False
    )

    all_metrics = []

    print("Running validation...")
    print()

    with torch.no_grad():

        for i in range(len(dataset)):

            image, mask = dataset[i]

            image_input = image.unsqueeze(0).to(DEVICE)

            logits = model(image_input)

            probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

            prediction = probability > 0.5

            target = mask[0].numpy() > 0.5

            dice, iou, precision, recall = calculate_metrics(
                prediction,
                target
            )

            all_metrics.append(
                [dice, iou, precision, recall]
            )

            print(
                f"{i+1:02d}/{len(dataset)} "
                f"Dice={dice:.4f} "
                f"IoU={iou:.4f} "
                f"Precision={precision:.4f} "
                f"Recall={recall:.4f}"
            )

    metrics = np.array(all_metrics)

    mean_metrics = metrics.mean(axis=0)

    print()
    print("=" * 70)
    print("INDEPENDENT VALIDATION RESULTS")
    print("=" * 70)

    print(f"Dice:      {mean_metrics[0]:.4f}")
    print(f"IoU:       {mean_metrics[1]:.4f}")
    print(f"Precision: {mean_metrics[2]:.4f}")
    print(f"Recall:    {mean_metrics[3]:.4f}")

    print("=" * 70)


if __name__ == "__main__":
    main()