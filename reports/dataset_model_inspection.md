# Dataset and Model Inspection Report

**Project:** Athena Antarctic Navigation (Athena-CV)  
**Date:** September 13, 2026  
**Inspection Scope:** Data Inventory, SAR GeoTIFFs, SIC NetCDFs, Auxiliary Layers, Label Presence, Pretrained Checkpoint  

---

## 1. Data Inventory

| Directory / File | File Type / Format | Size | Details / Description |
| :--- | :--- | :--- | :--- |
| `models/resunet_v1.0.0.pth` | PyTorch Checkpoint (ZIP / Pickle) | 104 MB | Pretrained ResUNet + CBAM model state dictionary |
| `data/demo/TIF/S1B__EW___A_20210214T104842_HV_grd_mli_gamma0-rtc_geo_db_3031.tif` | GeoTIFF (Float32) | 5.5 MB | Sentinel-1B EW SAR image (HV pol, RTC dB, EPSG:3031) |
| `data/demo/TIF/S1B__EW___A_20210310T104842_HV_grd_mli_gamma0-rtc_geo_db_3031.tif` | GeoTIFF (Float32) | 5.5 MB | Sentinel-1B EW SAR image (HV pol, RTC dB, EPSG:3031) |
| `data/demo/TIF/S1B__EW___A_20210415T104843_HV_grd_mli_gamma0-rtc_geo_db_3031.tif` | GeoTIFF (Float32) | 5.5 MB | Sentinel-1B EW SAR image (HV pol, RTC dB, EPSG:3031) |
| `data/demo/TIF/S1B__EW___A_20210427T104843_HV_grd_mli_gamma0-rtc_geo_db_3031.tif` | GeoTIFF (Float32) | 5.5 MB | Sentinel-1B EW SAR image (HV pol, RTC dB, EPSG:3031) |
| `data/demo/SIC/ice_conc_sh_polstere-100_amsr2_202102141200.nc` | NetCDF-4 / HDF5 | 52 KB | OSI SAF AMSR2 Sea Ice Concentration (2021-02-14) |
| `data/demo/SIC/ice_conc_sh_polstere-100_amsr2_202103101200.nc` | NetCDF-4 / HDF5 | 52 KB | OSI SAF AMSR2 Sea Ice Concentration (2021-03-10) |
| `data/demo/SIC/ice_conc_sh_polstere-100_amsr2_202104151200.nc` | NetCDF-4 / HDF5 | 52 KB | OSI SAF AMSR2 Sea Ice Concentration (2021-04-15) |
| `data/demo/SIC/ice_conc_sh_polstere-100_amsr2_202104271200.nc` | NetCDF-4 / HDF5 | 52 KB | OSI SAF AMSR2 Sea Ice Concentration (2021-04-27) |
| `data/demo/auxiliary/mini_fast_ice.gpkg` | GeoPackage (SQLite vector) | 48 KB | Fast ice vector polygon boundary |
| `data/demo/auxiliary/mini_coastline.gpkg` | GeoPackage (SQLite vector) | 44 KB | Coastline vector layer (empty schema) |
| `data/demo/auxiliary/mini_ibcso_bed.tif` | GeoTIFF (Int16) | 18 KB | IBCSO Bathymetric bed elevation grid |

---

## 2. SAR Properties

- **File Count:** 4 GeoTIFF files (`S1B__EW___A_20210214...tif`, `20210310...tif`, `20210415...tif`, `20210427...tif`)
- **Sensor & Processing:** Sentinel-1B, Extra Wide Swath (EW), Ground Range Detected (GRD), Medium Resolution, HV Polarization, Radiometrically Terrain Corrected (RTC) Gamma0, Decibel (dB) scale.
- **Dimensions:** 1,250 x 1,090 pixels (Width x Height).
- **Bands:** 1 band (32-bit Floating Point / Float32).
- **Pixel Resolution:** 40.0 m x 40.0 m per pixel.
- **Coordinate Reference System (CRS):** `EPSG:3031` (WGS 84 / Antarctic Polar Stereographic).
- **Bounding Box (EPSG:3031 Extent):**
  - $X_{\text{min}} = 1,820,000.0\text{ m}$
  - $X_{\text{max}} = 1,870,000.0\text{ m}$ (Width: 50.0 km)
  - $Y_{\text{min}} = -1,939,760.0\text{ m}$
  - $Y_{\text{max}} = -1,896,160.0\text{ m}$ (Height: 43.6 km)
- **Temporal Span:** 2021-02-14 to 2021-04-27 (4 distinct acquisition timestamps).

---

## 3. SIC Properties (Representative File: `ice_conc_sh_polstere-100_amsr2_202102141200.nc`)

- **Dataset / Source:** EUMETSAT OSI SAF AMSR2 Southern Hemisphere Sea Ice Concentration (`polstere-100` grid).
- **File Format:** NetCDF-4 / HDF5.
- **Primary Variables:**
  - `ice_conc`: Sea Ice Concentration on Southern Hemisphere
    - `long_name`: "The sea ice concentration on the southern hemisphere"
    - `standard_name`: `sea_ice_area_fraction`
    - `units`: `%`
    - `scale_factor`: `0.01`
    - `add_offset`: `0.0`
    - `valid_min`: `0`
    - `valid_max`: `10000` (represents 100.0%)
    - `_FillValue`: `-9990`
  - `status_flag`: Retrieval status flag (values: 2 = lake, 14 = ice mask, 100 = land, 101 = missing).
  - `total_uncertainty`, `smearing_uncertainty`, `algorithm_uncertainty`: Uncertainty fields (`units: %`, `scale_factor: 0.01`).
  - `xc`, `yc`: Spatial projection coordinates (`units: km`, `standard_name: projection_x_coordinate / projection_y_coordinate`).
  - `lat`, `lon`: Latitude and Longitude grids (`degrees_north`, `degrees_east`).
  - `time`: Time coordinate (`seconds since 1978-01-01 00:00:00`).
- **Grid Mapping / Projection:** Polar Stereographic (`Polar_Stereographic_Grid`)
  - PROJ string: `+proj=stere +a=6378273 +b=6356889.44891 +lat_0=-90 +lat_ts=-70 +lon_0=0` (Standard OSI SAF Southern Hemisphere 10 km grid).

---

## 4. Auxiliary Data

1. **`mini_fast_ice.gpkg`**
   - **Format:** GeoPackage vector database.
   - **Table Name:** `mini_fast_ice`
   - **CRS:** `EPSG:3031` (WGS 84 / Antarctic Polar Stereographic).
   - **Geometry Type:** `POLYGON` (1 feature, type: `'Fast Ice (Coast Included)'`).
   - **Spatial Extent:** Exactly matches the SAR bounding box: $X \in [1820000, 1870000]\text{ m}, Y \in [-1939760, -1896160]\text{ m}$.

2. **`mini_coastline.gpkg`**
   - **Format:** GeoPackage vector database.
   - **Table Name:** `mini_coastline`
   - **CRS:** `EPSG:3031` (WGS 84 / Antarctic Polar Stereographic).
   - **Geometry Type:** `GEOMETRY` (Feature count: 0, table is empty).

3. **`mini_ibcso_bed.tif`**
   - **Format:** GeoTIFF raster (16-bit signed integer / Int16).
   - **Data Type:** IBCSO (International Bathymetric Chart of the Southern Ocean) Bed Elevation.
   - **Dimensions:** 99 x 87 pixels (Width x Height).
   - **Resolution:** 500.0 m x 500.0 m.
   - **Bands:** 1 band.
   - **CRS:** WGS 84 / IBCSO Polar Stereographic (`EPSG:9329` / `EPSG:3031` derivative).
   - **Bounding Box:** $X \in [1782981.45, 1832481.45]\text{ m}, Y \in [-1900981.45, -1857481.45]\text{ m}$ (partially overlaps SAR extent with spatial shift).

---

## 5. Label Presence Check

- **Iceberg Masks / Annotations:** **NONE EXIST LOCALLY**.
- **Search Result:** Inspection of all 14 files in the workspace confirms there are no ground truth segmentation masks, iceberg bounding boxes, or annotated label files in `data/demo/` or elsewhere in the project.

---

## 6. Pretrained Model Information (`models/resunet_v1.0.0.pth`)

- **File Size:** 104 MB (109,036,547 bytes).
- **Format:** PyTorch ZIP archive containing pickled state dictionary (`unet_tif_hv_1536.2.2`).
- **Architecture:** Residual U-Net (ResUNet) enhanced with Convolutional Block Attention Module (CBAM) blocks.
- **Encoder Backbone:** ResNet-152 structure ($3 + 4 + 36 + 3$ Bottleneck blocks):
  - `encoder.0`: Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
  - `encoder.4`: 3 Bottleneck blocks (64 -> 256 channels)
  - `encoder.5`: 4 Bottleneck blocks (256 -> 512 channels)
  - `encoder.6`: 36 Bottleneck blocks (512 -> 1024 channels)
  - `encoder.7`: 3 Bottleneck blocks (1024 -> 2048 channels)
- **Attention Modules:** CBAM (Channel Attention + Spatial Attention with 7x7 conv) integrated on skip connections (`cbam1`, `cbam2`, `cbam3`, `cbam4`) and bottleneck bridge (`cbam_center`).
- **Decoder Structure:**
  - `decoder4`: Conv2d(3072, 512, 3x3) -> Conv2d(512, 512, 3x3)
  - `decoder3`: Conv2d(1024, 256, 3x3) -> Conv2d(256, 256, 3x3)
  - `decoder2`: Conv2d(512, 128, 3x3) -> Conv2d(128, 128, 3x3)
  - `decoder1`: Conv2d(192, 64, 3x3) -> Conv2d(64, 64, 3x3)
  - `output`: Conv2d(64, 1, 1x1)
- **Input Channels:** **1 channel** (`encoder.0.weight` shape: `[64, 1, 7, 7]`). Designed for single-channel SAR imagery (e.g., Sentinel-1 HV polarization).
- **Output Channels:** **1 channel** (`output.weight` shape: `[1, 64, 1, 1]`). Designed for single-channel continuous/binary probability or segmentation mask map.
- **Internal Tag:** `unet_tif_hv_1536.2.2` (indicates training on SAR HV GeoTIFFs, likely 1536x1536 patch size).

---

## 7. Important Unknowns

1. **Specific Prediction Objective / Task:** While the model accepts 1 channel and outputs 1 channel, it is unverified whether it was trained specifically for **iceberg detection/segmentation**, **sea ice concentration estimation**, or **fast ice segmentation**.
2. **Missing Ground Truth Labels:** No local ground truth iceberg masks or sea ice target labels exist in `data/demo/`. Evaluation or fine-tuning cannot occur locally without acquiring label datasets.
3. **Data Normalization & Clipping:** The exact intensity normalization applied to input SAR HV dB values (e.g., min-max scaling between -35 dB and 0 dB vs. z-score standardization) is not saved inside the PyTorch state dict.
4. **Spatial Alignment of Auxiliary Datasets:**
   - `mini_ibcso_bed.tif` is on a 500m grid with a spatial offset relative to the 40m SAR grid.
   - `mini_coastline.gpkg` is empty (0 features).

---

## 8. Recommended Next Steps

1. **Construct Spatial Data Pipeline:** Build a lightweight raster processing module to reproject and align auxiliary vectors (`mini_fast_ice.gpkg`) and rasters (`mini_ibcso_bed.tif`) onto the SAR 40m EPSG:3031 grid.
2. **Define ResUNet-152-CBAM PyTorch Model Definition:** Reconstruct the PyTorch `nn.Module` class matching `resunet_v1.0.0.pth` layer names to allow loading the state dict cleanly.
3. **Establish Input Preprocessing Standard:** Standardize SAR HV dB image normalization and patch tiling strategy (e.g., 512x512 or 1536x1536 sliding window tiles).
4. **Acquire Ground Truth Labels:** If iceberg detection or segmentation evaluation is required, obtain annotated iceberg masks or ground truth polygon labels.
