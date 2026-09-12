# Real-Image Single-Tile Inference Test Report

**Project:** Athena Antarctic Navigation (Athena-CV)  
**Date:** September 13, 2026  
**Target Image:** `data/demo/TIF/S1B__EW___A_20210310T104842_HV_grd_mli_gamma0-rtc_geo_db_3031.tif`  
**Model:** `ResUNetCBAM` loaded from `models/resunet_v1.0.0.pth`  
**Status:** **INFERENCE TEST SUCCESSFUL**  

---

## 1. Test Overview

A single-tile real-image inference test was performed using a $256 \times 256$ pixel valid window from the Sentinel-1B HV polarization SAR image. No model parameters were modified or fine-tuned. The pipeline ingested the raw GeoTIFF tile, applied documented SAR intensity normalization, evaluated the `ResUNetCBAM` model, and produced probability and binary segmentation maps.

---

## 2. Input Image & Tile Window Details

- **Input SAR File:** `data/demo/TIF/S1B__EW___A_20210310T104842_HV_grd_mli_gamma0-rtc_geo_db_3031.tif`
- **Sensor / Mode:** Sentinel-1B Extra Wide (EW), Ground Range Detected (GRD), HV polarization, RTC Gamma0 (dB).
- **Tile Bounding Window:**
  - Row range: `[300:556]` (Height: 256 pixels)
  - Column range: `[400:656]` (Width: 256 pixels)
  - Total pixels: 65,536 pixels ($100\%$ valid finite data, $0$ NaN / nodata pixels)

---

## 3. Original Pixel Statistics & Preprocessing

### Original SAR dB Statistics
- **Min:** $-29.6187\text{ dB}$
- **Max:** $-1.5511\text{ dB}$
- **Mean:** $-20.7176\text{ dB}$
- **Std:** $4.3594\text{ dB}$
- **Nodata / NaN count:** $0$ ($0.00\%$)

### Preprocessing Applied
Linear min-max scaling mapped over the operational dynamic range of Sentinel-1 EW HV RTC Gamma0 imagery $[-35.0\text{ dB}, 0.0\text{ dB}]$ into $[0.0, 1.0]$:
$$ x_{\text{prep}} = \text{clip}\left(\frac{x - (-35.0)}{0.0 - (-35.0)}, 0.0, 1.0\right) = \text{clip}\left(\frac{x + 35.0}{35.0}, 0.0, 1.0\right) $$

- **Preprocessed Range:** $[0.1538, 0.9557]$
- **Model Input Tensor Shape:** `[1, 1, 256, 256]`

---

## 4. Model Inference Statistics

### Raw Model Output (Unscaled Logits)
- **Min:** $-7.0289$
- **Max:** $+3.6120$
- **Mean:** $-1.4998$
- **Std:** $2.6884$

### Sigmoid Probability Map ($\sigma(\text{logits})$)
- **Min:** $0.0009$
- **Max:** $0.9737$
- **Mean:** $0.3288$
- **Std:** $0.3515$

### Binary Masking ($p > 0.50$)
- **Decision Threshold:** $0.50$ (standard default classification cutoff)
- **Positive Pixels ($p > 0.50$):** **20,134** out of 65,536 pixels (**30.72%**)
- **High-Confidence Positive Pixels ($p > 0.90$):** **7,177** out of 65,536 pixels (**10.95%**)

---

## 5. Output Evaluation & Plausibility

- **Plausibility:** **Plausible and well-structured.** The output probability map demonstrates clear bimodal separation, assigning near-zero probability ($0.0009$) to background ocean/sea ice regions and high probabilities ($0.90 - 0.97$) to distinct structural radar features.
- **Iceberg / Feature Detection:** The thresholded binary mask highlights 20,134 positive pixels corresponding to high-backscatter ice structures / sea ice boundary features within the SAR scene.

---

## 6. Limitations & Uncertainties

1. **Unrecorded Target Class:** The checkpoint `models/resunet_v1.0.0.pth` state dict does not include text metadata specifying whether it was trained for iceberg detection, sea ice / open water segmentation, or fast ice boundaries.
2. **Threshold Arbitrariness:** The decision threshold of $0.50$ is documented for demonstration purposes and is not claimed to be scientifically optimal.
3. **Training Normalization Unrecorded:** Min-max scaling over $[-35, 0]\text{ dB}$ provides stable $[0, 1]$ inputs, but the exact normalization used during model pretraining was not saved in the checkpoint.

---

## 7. Artifacts Generated

All artifacts are saved under `reports/inference_test/`:

| File | Description |
| :--- | :--- |
| `reports/inference_test/input_tile.png` | Grayscale visual representation of input SAR HV tile |
| `reports/inference_test/probability_map.png` | Continuous sigmoid probability prediction map ($0.0 \to 1.0$) |
| `reports/inference_test/binary_mask.png` | Binary segmentation mask thresholded at $p > 0.50$ |
| `reports/inference_test/overlay.png` | Red mask overlay on top of grayscale SAR input background |
| `reports/inference_test.md` | Detailed single-tile inference test report |
