import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel

IMG = 448                    # input size, same as the experiments
PATCH = 14                   # DINOv2 patch size
GRID = IMG // PATCH          # 32 x 32 patch grid
MAP_SIZE = 256               # maps are computed at 256x256, like the thresholds were

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_transform = T.Compose([
    T.Resize((IMG, IMG)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model():
    return AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()


@torch.no_grad()
def patch_features(model, image):
    """One L2-normalised feature vector per 14x14 patch: shape (1024, 768)."""
    x = _transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    patches = model(pixel_values=x).last_hidden_state[0, 1:]     # drop the CLS token
    return patches / patches.norm(dim=-1, keepdim=True) 


def build_bank(model, ref_dir):
    """Memory bank = patch features of all normal reference images in ref_dir."""
    files = sorted(f for f in os.listdir(ref_dir) if f.lower().endswith(".png"))
    if not files:
        raise FileNotFoundError(f"No reference images found in {ref_dir}")
    feats = [patch_features(model, Image.open(os.path.join(ref_dir, f))) for f in files]
    return torch.cat(feats, dim=0)                               # (n_images * 1024, 768)


@torch.no_grad()
def anomaly_map(model, image, bank):
    """256x256 map: high where a patch has no close match among the normal references."""
    nearest = (patch_features(model, image) @ bank.T).max(dim=1).values
    score = (1 - nearest).reshape(GRID, GRID).cpu().numpy()
    score = cv2.resize(score, (MAP_SIZE, MAP_SIZE), interpolation=cv2.INTER_LINEAR)  
    return cv2.GaussianBlur(score, (0, 0), sigmaX=4)
