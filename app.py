import json
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import dino_matcher as dm

HERE = os.path.dirname(os.path.abspath(__file__))
THRESHOLDS = json.load(open(os.path.join(HERE, "thresholds.json")))

# Measured in our study on the MVTec AD test sets: DINOv2 448, K=10, seed 0
BENCHMARK = {
    "carpet":   {"Image AUROC": 0.999, "Pixel AUROC": 0.991, "PRO": 0.972},
    "grid":     {"Image AUROC": 1.000, "Pixel AUROC": 0.994, "PRO": 0.971},
    "bottle":   {"Image AUROC": 1.000, "Pixel AUROC": 0.989, "PRO": 0.963},
    "hazelnut": {"Image AUROC": 0.996, "Pixel AUROC": 0.995, "PRO": 0.972},
}


@st.cache_resource(show_spinner="Loading DINOv2 (the first run downloads about 350 MB)...")
def get_model():
    return dm.load_model()


@st.cache_resource(show_spinner="Building the reference memory bank...")
def get_bank(category):
    return dm.build_bank(get_model(), os.path.join(HERE, "reference", category)) 


def render(image, amap, thr):
    """Returns (heatmap overlay, outlined defect region), both as RGB arrays at image size."""
    rgb = np.array(image)
    h, w = rgb.shape[:2]
    big = cv2.resize(amap, (w, h), interpolation=cv2.INTER_LINEAR)

    # Fixed colour scale (0 to 2x threshold), so colours mean the same on every image
    norm = np.clip(big / (2 * thr), 0, 1)
    colour = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb, 0.5, colour, 0.5, 0)

    mask = (big > thr).astype(np.uint8)
    outlined = rgb.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(outlined, contours, -1, (255, 0, 0), max(2, w // 200))
    return overlay, outlined, float(mask.mean())


st.set_page_config(page_title="Zero-shot defect detection", layout="wide")
st.title("Industrial defect detection with DINOv2 patch matching")
st.caption("Uses 10 normal reference images per category and no defect examples. "
           "Research demo, not a production inspection system.")

category = st.sidebar.selectbox("Product category", list(THRESHOLDS))
st.sidebar.write("Pick the category that matches your image. The app cannot check this.")

sample_dir = os.path.join(HERE, "samples", category)
samples = sorted(os.listdir(sample_dir)) if os.path.isdir(sample_dir) else []
source = st.sidebar.radio("Image source", ["Sample image", "Upload my own"])

image = None
if source == "Sample image" and samples:
    choice = st.sidebar.selectbox("Sample", samples)
    image = Image.open(os.path.join(sample_dir, choice))
else:
    uploaded = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(uploaded)

if image is None:
    st.info("Choose a sample or upload an image in the sidebar.")
    st.stop()

image = image.convert("RGB")
amap = dm.anomaly_map(get_model(), image, get_bank(category))
thr = THRESHOLDS[category]["threshold"]
score = float(amap.max())
overlay, outlined, flagged = render(image, amap, thr)

c1, c2, c3 = st.columns(3)
c1.image(image, caption="Input image", use_container_width=True)
c2.image(overlay, caption="Anomaly heatmap (fixed colour scale)", use_container_width=True)
c3.image(outlined, caption="Flagged region (red outline)", use_container_width=True)

m1, m2, m3, m4 = st.columns(4)
if score > thr:
    m1.error("ANOMALY")
else:
    m1.success("NORMAL")
m2.metric("Anomaly score", f"{score:.3f}")
m3.metric("Decision threshold", f"{thr:.3f}")
m4.metric("Flagged area", f"{flagged * 100:.1f}%")

if flagged > 0.5:
    st.warning(
        "More than half of the image is flagged. This usually means the image does not look like "
        "the reference images (different product, camera angle or lighting), so this result is unreliable."
    )

with st.expander("What these numbers mean, and what this cannot do"):
    st.markdown(
        "- The score is the largest patch distance to the normal references. It is **not a probability or a confidence**.\n"
        "- The threshold is the highest score among 20 held-out normal training images of this category. No test labels were used to set it.\n"
        "- Works only for the four categories we tested, on images shot like the MVTec AD ones. "
        "Different lighting, camera or product will change the scores and the threshold would no longer be valid.\n"
        "- The benchmark below is the study's result on the MVTec AD test sets. It says nothing about the image above."
    )
    st.table(BENCHMARK[category])
