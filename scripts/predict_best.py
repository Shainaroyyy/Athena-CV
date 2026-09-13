from pathlib import Path
import sys

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import UNet, IcebergDataset, test_pairs, DEVICE


MODEL_PATH = Path("reports/training/best_unet.pth")
OUTPUT_DIR = Path("reports/training/best_predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("BEST MODEL PREDICTION")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Test samples: {len(test_pairs)}")

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
        test_pairs,
        training=False
    )

    print()
    print("Generating predictions...")

    with torch.no_grad():
        for i in range(min(5, len(dataset))):

            image, mask = dataset[i]

            image_input = image.unsqueeze(0).to(DEVICE)

            logits = model(image_input)
            probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

            prediction = (
                probability > 0.5
            ).astype(np.uint8) * 255

            ground_truth = (
                mask[0].numpy() * 255
            ).astype(np.uint8)

            sar = (
                image[0].numpy() * 255
            ).clip(0, 255).astype(np.uint8)

            Image.fromarray(sar).save(
                OUTPUT_DIR / f"{i:02d}_sar.png"
            )

            Image.fromarray(ground_truth).save(
                OUTPUT_DIR / f"{i:02d}_ground_truth.png"
            )

            Image.fromarray(prediction).save(
                OUTPUT_DIR / f"{i:02d}_prediction.png"
            )

            print(f"  ✓ Sample {i:02d} saved")

    print()
    print("=" * 70)
    print(f"Done. Results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()