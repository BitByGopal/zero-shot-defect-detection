# Zero-Shot Industrial Defect Detection (Demo)

A Streamlit demo of reference-based patch matching with DINOv2 for finding and locating defects on MVTec AD. It is part of my individual research project **"Zero-Shot Industrial Defect Detection Using Foundation Vision Models"** (B.Tech CSE AI/ML, Lovely Professional University).

**Live demo:** https://zero-shot-defect-detection-gollagopal.streamlit.app/

## What it does

Upload an image (or pick a sample), choose a category, and the app shows:
- the anomaly heatmap on a fixed colour scale,
- a red outline around the flagged region,
- the decision (NORMAL / ANOMALY), the anomaly score, the threshold and the flagged-area percentage,
- a warning when more than half the image is flagged, which usually means the input does not look like the reference images.

Categories: carpet, grid, bottle, hazelnut.

## How it works

1. Patch features come from `facebook/dinov2-base` at 448 px (32x32 patch grid).
2. A memory bank is built from 10 normal reference images per category (`reference/`).
3. Each test patch gets the score `1 - max cosine similarity` to the bank.
4. The map is resized to 256x256 and smoothed with a Gaussian filter (sigma 4). The image score is the maximum of the map.
5. The threshold per category (`thresholds.json`) is the highest score over 20 held-out normal training images. No defective or test image is used to set it.

This is zero-shot for defects but few-shot for normal appearance, because it uses normal reference images. The idea follows PatchCore and AnomalyDINO. It is **not** a new method.

## Results from the study (MVTec AD, DINOv2 448 px, k = 10, seed 0)

| Category | Image AUROC | Pixel AUROC | PRO |
|---|---|---|---|
| Carpet | 0.999 | 0.991 | 0.972 |
| Grid | 1.000 | 0.994 | 0.971 |
| Bottle | 1.000 | 0.989 | 0.963 |
| Hazelnut | 0.996 | 0.995 | 0.972 |
| **Mean** | **0.999** | **0.992** | **0.969** |

Limits: four categories only, one headline seed (five-seed stability was checked), threshold from a single 20-image draw. A text-only CLIP baseline in the same study reached a mean image AUROC of 0.893. The full comparison is in the report.

## Run locally

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the DINOv2 weights. CPU inference takes a few seconds per image.

## Files

```
app.py            Streamlit interface
dino_matcher.py   DINOv2 features, memory bank, anomaly map
thresholds.json   per-category thresholds
reference/        10 normal reference images per category
samples/          example test images
requirements.txt
```

## Data and credits

The images in `reference/` and `samples/` are from **MVTec AD** (Bergmann et al., CVPR 2019 and IJCV 2021), used here for non-commercial research and demonstration only. Check the dataset licence at https://www.mvtec.com/company/research/datasets/mvtec-ad before reusing them.

Related work: PatchCore (Roth et al., CVPR 2022), AnomalyDINO (Damm et al., WACV 2025), DINOv2 (Oquab et al., TMLR 2024), WinCLIP (Jeong et al., CVPR 2023).

## Disclaimer

Research demo. Not validated for production inspection.