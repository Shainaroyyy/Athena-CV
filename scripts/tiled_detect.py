
from pathlib import Path
import sys
import json

import numpy as np
import torch
import rasterio
from PIL import Image, ImageDraw
from scipy import ndimage
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import UNet, resize_image, DEVICE


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path("reports/training/best_unet.pth")

TILE_H = 239
TILE_W = 250

STRIDE_H = 120
STRIDE_W = 125

BATCH_SIZE = 16
THRESHOLD = 0.5
MIN_COMPONENT_PIXELS = 5


# ============================================================
# NORMALIZATION
# Same logic as training load_sar()
# ============================================================

def normalize_tile(image):

    image = image.astype(np.float32)

    valid = np.isfinite(image)

    if not valid.any():
        return np.zeros_like(image, dtype=np.float32)

    values = image[valid]

    lo, hi = np.percentile(values, [1, 99])

    if hi <= lo:
        lo = values.min()
        hi = values.max()

    image = np.clip(image, lo, hi)
    image = (image - lo) / (hi - lo + 1e-8)

    image[~valid] = 0.0

    return image.astype(np.float32)


# ============================================================
# TILE GENERATION
# ============================================================

def generate_windows(height, width):

    rows = list(range(0, max(height - TILE_H, 0) + 1, STRIDE_H))
    cols = list(range(0, max(width - TILE_W, 0) + 1, STRIDE_W))

    if rows[-1] != height - TILE_H:
        rows.append(max(height - TILE_H, 0))

    if cols[-1] != width - TILE_W:
        cols.append(max(width - TILE_W, 0))

    windows = []

    for r in rows:
        for c in cols:
            windows.append((r, c))

    return windows


# ============================================================
# MODEL
# ============================================================

def load_model():

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

    return model


# ============================================================
# TILE INFERENCE
# ============================================================

def tiled_probability(image, model):

    height, width = image.shape

    probability_sum = np.zeros((height, width), dtype=np.float32)
    probability_count = np.zeros((height, width), dtype=np.float32)

    windows = generate_windows(height, width)

    print(f"Tiles: {len(windows)}")

    for start in range(0, len(windows), BATCH_SIZE):

        batch_windows = windows[start:start + BATCH_SIZE]

        batch = []

        for r, c in batch_windows:

            tile = image[r:r + TILE_H, c:c + TILE_W]

            if tile.shape != (TILE_H, TILE_W):

                padded = np.zeros((TILE_H, TILE_W), dtype=np.float32)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded

            tile = normalize_tile(tile)
            tile = resize_image(tile)

            batch.append(tile)

        batch_tensor = torch.from_numpy(
            np.stack(batch)
        ).float().unsqueeze(1).to(DEVICE)

        with torch.no_grad():

            logits = model(batch_tensor)

            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]

        for idx, (r, c) in enumerate(batch_windows):

            prob = probs[idx]

            restored = Image.fromarray(prob.astype(np.float32), mode="F")

            restored = restored.resize(
                (TILE_W, TILE_H),
                Image.Resampling.BILINEAR
            )

            restored = np.array(restored, dtype=np.float32)

            actual_h = min(TILE_H, height - r)
            actual_w = min(TILE_W, width - c)

            probability_sum[
                r:r + actual_h,
                c:c + actual_w
            ] += restored[:actual_h, :actual_w]

            probability_count[
                r:r + actual_h,
                c:c + actual_w
            ] += 1

        print(
            f"Processed {min(start + BATCH_SIZE, len(windows))}/{len(windows)} tiles"
        )

    probability = probability_sum / np.maximum(probability_count, 1)

    return probability


# ============================================================
# GEOSPATIAL
# ============================================================

def extract_objects(probability, transform, crs):

    binary = probability > THRESHOLD

    structure = np.ones((3, 3), dtype=np.uint8)

    labels, count = ndimage.label(
        binary,
        structure=structure
    )

    transformer = Transformer.from_crs(
        crs,
        "EPSG:4326",
        always_xy=True
    )

    pixel_area = abs(transform.a * transform.e)

    objects = []

    for component_id in range(1, count + 1):

        component = labels == component_id

        pixel_count = int(component.sum())

        if pixel_count < MIN_COMPONENT_PIXELS:
            continue

        rows, cols = np.where(component)

        centroid_row = float(rows.mean())
        centroid_col = float(cols.mean())

        x, y = rasterio.transform.xy(
            transform,
            centroid_row,
            centroid_col,
            offset="center"
        )

        lon, lat = transformer.transform(x, y)

        area_km2 = (
            pixel_count * pixel_area
        ) / 1_000_000

        confidence = float(
            probability[component].mean()
        )

        objects.append({
            "iceberg_id": "",
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "area_km2": round(area_km2, 6),
            "confidence": round(confidence, 4),
            "pixel_count": pixel_count
        })

    objects.sort(
        key=lambda x: x["area_km2"],
        reverse=True
    )

    for i, obj in enumerate(objects, 1):
        obj["iceberg_id"] = f"IB_{i:03d}"

    return binary, labels, objects


# ============================================================
# VISUALIZATION
# ============================================================

def create_visualization(image, labels, objects, output):

    img = normalize_tile(image)

    rgb = Image.fromarray(
        (img * 255).astype(np.uint8)
    ).convert("RGB")

    draw = ImageDraw.Draw(rgb)

    for idx, obj in enumerate(objects, 1):

        component = labels == np.unique(labels)[
            np.unique(labels) > 0
        ][idx - 1] if idx <= len(np.unique(labels)) - 1 else None

        if component is None:
            continue

        rows, cols = np.where(component)

        if len(rows) == 0:
            continue

        x1 = cols.min()
        y1 = rows.min()
        x2 = cols.max()
        y2 = rows.max()

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(255, 0, 0),
            width=2
        )

        draw.text(
            (int(cols.mean()), int(rows.mean())),
            str(idx),
            fill=(255, 255, 0)
        )

    rgb.save(output)


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 2:

        print("Usage:")
        print("python3 scripts/tiled_detect.py <image.tif>")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    with rasterio.open(image_path) as src:

        image = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs

    print("=" * 70)
    print("ATHENA CV — TILED ICEBERG DETECTION")
    print("=" * 70)
    print(f"Input: {image_path.name}")
    print(f"Original size: {image.shape}")
    print(f"Device: {DEVICE}")
    print()

    model = load_model()

    probability = tiled_probability(image, model)

    binary, labels, objects = extract_objects(
        probability,
        transform,
        crs
    )

    output_dir = Path("reports/tiled")
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem

    Image.fromarray(
        (probability * 255).astype(np.uint8)
    ).save(output_dir / f"{stem}_probability.png")

    Image.fromarray(
        (binary * 255).astype(np.uint8)
    ).save(output_dir / f"{stem}_mask.png")

    json_path = output_dir / f"{stem}_icebergs.json"

    with open(json_path, "w") as f:
        json.dump(
            {
                "image": image_path.name,
                "iceberg_count": len(objects),
                "icebergs": objects
            },
            f,
            indent=2
        )

    vis_path = output_dir / f"{stem}_detections.png"

    create_visualization(
        image,
        labels,
        objects,
        vis_path
    )

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Icebergs detected: {len(objects)}")
    print(f"JSON: {json_path}")
    print(f"Visualization: {vis_path}")


if __name__ == "__main__":
    main()