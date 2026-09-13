from pathlib import Path
from PIL import Image
import rasterio
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("data/labeled")
OUT = Path("reports/dataset_inspection")
OUT.mkdir(parents=True, exist_ok=True)


def inspect(name, image_dir, mask_dir):
    images = sorted(image_dir.glob("*.tif"))
    masks = sorted(mask_dir.glob("*.png"))

    # Match by filename, removing "_mask" from mask names
    mask_map = {m.stem.replace("_mask", ""): m for m in masks}

    pairs = [(img, mask_map[img.stem])
             for img in images if img.stem in mask_map]

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(f"Images : {len(images)}")
    print(f"Masks  : {len(masks)}")
    print(f"Pairs  : {len(pairs)}")

    if not pairs:
        print("NO MATCHED PAIRS")
        return

    # Inspect first actual matched pair
    image_path, mask_path = pairs[0]

    with rasterio.open(image_path) as src:
        image = src.read(1)
        print("\nIMAGE")
        print("File       :", image_path.name)
        print("Shape      :", image.shape)
        print("Dtype      :", image.dtype)
        print("Min/Max    :", np.nanmin(image), np.nanmax(image))
        print("Mean/Std   :", np.nanmean(image), np.nanstd(image))
        print("CRS        :", src.crs)
        print("Resolution :", src.res)

    mask = np.array(Image.open(mask_path))

    values, counts = np.unique(mask, return_counts=True)

    print("\nMASK")
    print("File       :", mask_path.name)
    print("Shape      :", mask.shape)
    print("Values     :", values.tolist())
    print("Counts     :", counts.tolist())

    # Visualize
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Sentinel-1 SAR")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")

    plt.tight_layout()

    output = OUT / f"{name.lower().replace(' ', '_')}_sample.png"
    plt.savefig(output, dpi=150)
    plt.close()

    print("Visualization:", output)


def main():

    inspect(
        "TRAIN",
        ROOT / "Train_Dataset" / "Train_TIF",
        ROOT / "Train_Dataset" / "Mask_PNG",
    )

    inspect(
        "TEST",
        ROOT / "Test_Dataset" / "Test_TIF",
        ROOT / "Test_Dataset" / "Mask_PNG",
    )

    inspect(
        "EXTENDED_VALIDATION",
        ROOT / "Validation_Dataset" / "Extended_Validation" / "TIF",
        ROOT / "Validation_Dataset" / "Extended_Validation" / "Mask_PNG",
    )

    inspect(
        "MONTHLY_VALIDATION",
        ROOT / "Validation_Dataset" / "Monthly_Validation_Adelie" / "TIF",
        ROOT / "Validation_Dataset" / "Monthly_Validation_Adelie" / "Mask_PNG",
    )


if __name__ == "__main__":
    main()