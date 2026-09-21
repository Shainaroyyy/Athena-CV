# Athena-CV

### Antarctic Iceberg Detection from Sentinel-1 SAR Imagery

A computer vision pipeline for detecting and geolocating Antarctic icebergs from Sentinel-1 Synthetic Aperture Radar (SAR) imagery using deep learning and geospatial processing.

> **Project:** Athena — AI-Enabled Antarctic Navigation Decision Support System  
> **Module:** Computer Vision / Iceberg Detection

---

## Overview

Antarctic navigation is complicated by the presence and movement of icebergs and sea ice. Athena-CV focuses on the perception layer of this problem: extracting individual iceberg observations from satellite imagery that can subsequently be used by trajectory prediction and navigation modules.

The pipeline takes **georeferenced Sentinel-1 SAR imagery** as input and uses a **PyTorch U-Net** for pixel-level iceberg segmentation. The resulting segmentation mask is processed using connected-component analysis to extract individual iceberg candidates, which are then converted into geographic coordinates and physical area estimates.

### Pipeline

```text
Sentinel-1 SAR GeoTIFF
          │
          ▼
   Image Preprocessing
          │
          ▼
      Tiled Inference
          │
          ▼
     PyTorch U-Net
          │
          ▼
  Pixel-wise Probability Map
          │
          ▼
   Binary Segmentation Mask
          │
          ▼
 Connected Component Analysis
          │
          ▼
 Geospatial Transformation
          │
          ▼
Iceberg Detection JSON
