# Streamlit app for SkinBench: L1 (Normal/Abnormal) -> L2 (8 diseases) -> L3 (subclasses for Eczema/Fungal/Pox)
# - L1 uses ResNet-50 (2 classes)
# - L2 uses DenseNet-121 (8 classes, excluding "Normal")
# - L3 uses DenseNet-121 (6 subclasses: 2 under Eczema, 2 under Fungal, 2 under Pox)
# - Grad-CAM overlays shown for each step used
# - Threshold tau gates Normal vs Abnormal in L1

import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import streamlit as st
from matplotlib import cm
import pandas as pd

# ====== Import your repo models ======
from models.resnet_model import ResNet50
from models.DenseNet121 import DenseNet121Medical

device = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_TAU = 0.70

# ------------ Label spaces ------------
CLASSES_ALL9 = [
    "Acne", "Bacterial/Impetigo", "Eczema", "Fungal",
    "Normal", "Pigmentation", "Pox", "Psoriasis", "Scabies"
]
NORMAL_IDX = CLASSES_ALL9.index("Normal")
L2_LABELS = [c for c in CLASSES_ALL9 if c != "Normal"]

# Subclass labels (6 total) and routing map from L2→subset indices
SUBCLASS_LABELS = [
    "Atopic Dermatitis", "Seborrhoeic Dermatitis",   # Eczema (2)
    "Candidiasis", "Tinea",                          # Fungal (2)
    "Chickenpox", "Monkeypox"                        # Pox (2)
]
ROUTE_SUBSETS = {
    "Eczema":   {"idx": [0, 1], "title": "Eczema subclasses"},
    "Fungal":   {"idx": [2, 3], "title": "Fungal subclasses"},
    "Pox":      {"idx": [4, 5], "title": "Pox subclasses"},
}

# ------------ Grad-CAM helpers ------------
def _to_rgb01(x_tensor):
    x = x_tensor.detach().cpu().clone()
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    x = (x * std) + mean
    x = x.clamp(0, 1)
    return x.permute(1, 2, 0).numpy()   # HxWx3

def compute_gradcam(model, x, target_layer, target_index=None):
    """Return Grad-CAM heatmap (HxW in [0,1]) for one image (N=1)."""
    model.zero_grad(set_to_none=True)
    feats, grads = {}, {}

    def fwd_hook(_m, _i, o): feats["x"] = o
    def bwd_hook(_m, gi, go): grads["x"] = go[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    x = x.clone().requires_grad_(True)
    out = model(x)
    if target_index is None:
        target_index = out.argmax(1).item()
    out[:, target_index].sum().backward()

    g = grads["x"].detach()     # [N,C,H,W]
    f = feats["x"].detach()     # [N,C,H,W]
    w = g.mean(dim=(2,3), keepdim=True)
    cam = (w * f).sum(dim=1)    # [N,H,W]
    cam = cam.relu()
    cam -= cam.amin(dim=(1,2), keepdim=True)
    cam /= (cam.amax(dim=(1,2), keepdim=True) + 1e-6)
    cam = cam[0].cpu().numpy()

    h1.remove(); h2.remove()
    return cam

def overlay_cam(rgb_img_01, cam_01, alpha=0.35, cmap=cm.jet):
    from PIL import Image as PILImage
    h, w, _ = rgb_img_01.shape
    cam_img = PILImage.fromarray((cam_01 * 255).astype(np.uint8)).resize((w, h), PILImage.BILINEAR)
    cam_01 = np.asarray(cam_img).astype(np.float32) / 255.0
    cam_rgb = cmap(cam_01)[..., :3]
    out = (1 - alpha) * rgb_img_01 + alpha * cam_rgb
    return np.clip(out, 0, 1)

# ------------ Model loaders (cached) ------------
@st.cache_resource
def load_l1():
    model = ResNet50(num_classes=2).to(device)
    model.load_state_dict(torch.load("runs/L1/resnet50/best.pt", map_location=device))
    model.eval()
    target_layer = model.backbone.layer4
    return model, target_layer

@st.cache_resource
def load_l2():
    model = DenseNet121Medical(num_classes=len(L2_LABELS)).to(device)
    model.load_state_dict(torch.load("runs/L2/densenet121/best.pt", map_location=device))
    model.eval()
    target_layer = model.backbone.features.denseblock4
    return model, target_layer

@st.cache_resource
def load_l3():
    ckpt_path = "runs/SUBCLASS/densenet121/best.pt"   # <— changed
    model = DenseNet121Medical(num_classes=len(SUBCLASS_LABELS)).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))  # loads cleanly
    model.eval()
    target_layer = model.backbone.features.denseblock4
    return model, target_layer


l1, l1_t = load_l1()
l2, l2_t = load_l2()
l3, l3_t = load_l3()

# ------------ Preprocess ------------
TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ------------ UI ------------
st.set_page_config(page_title="SkinBench Multilayer (with Subclass) + Grad-CAM", layout="wide")
st.title("🧬 Skin Disease Detection — L1→L2→L3 (Subclass) with Grad-CAM")
st.caption("L1: Normal/Abnormal → L2: 8 diseases → L3: subclass (Eczema/Fungal/Pox).")

with st.sidebar:
    st.header("Settings")
    tau = st.slider("Threshold τ for L1 (Normal if p_normal ≥ τ)", 0.40, 0.90, float(DEFAULT_TAU), 0.01)
    cam_alpha = st.slider("CAM overlay α", 0.10, 0.80, 0.35, 0.05)
    st.markdown(f"**Device:** `{device}`")
    if l3 is None:
        st.warning("L3 checkpoint not found at `runs/SUBCLASS/subclass/best.pt`.\nSubclass step will be skipped.")

file = st.file_uploader("Upload a JPG/PNG image", type=["jpg", "jpeg", "png"])
if file is None:
    st.info("Upload an image to begin.")
    st.stop()

pil_img = Image.open(file).convert("RGB")
st.image(pil_img, caption="Uploaded image", use_container_width=True)

# ------------ Inference ------------
with st.spinner("Running inference…"):
    x = TRANSFORM(pil_img).unsqueeze(0).to(device)
    rgb = _to_rgb01(x[0])

    # L1
    with torch.inference_mode():
        p = torch.softmax(l1(x), dim=1)[0]  # [2]
    p_normal = float(p[0].item())
    l1_pred = "Normal" if p_normal >= tau else "Abnormal"
    l1_idx = 0 if l1_pred == "Normal" else 1
    l1_cam = compute_gradcam(l1, x, l1_t, target_index=l1_idx)
    l1_overlay = overlay_cam(rgb, l1_cam, alpha=cam_alpha)

    # L2 (only if abnormal)
    l2_pred_name, l2_conf, l2_overlay, q2 = None, None, None, None
    if l1_pred == "Abnormal":
        with torch.inference_mode():
            q2 = torch.softmax(l2(x), dim=1)[0].cpu().numpy()  # (8,)
        k2 = int(np.argmax(q2))
        l2_pred_name, l2_conf = L2_LABELS[k2], float(q2[k2])
        l2_cam = compute_gradcam(l2, x, l2_t, target_index=k2)
        l2_overlay = overlay_cam(rgb, l2_cam, alpha=cam_alpha)

    # L3 (subclass) if needed
    l3_pred_name, l3_conf, l3_overlay, q3_vis = None, None, None, None
    if l2_pred_name in ROUTE_SUBSETS and l3 is not None:
        with torch.inference_mode():
            q3 = torch.softmax(l3(x), dim=1)[0].cpu().numpy()  # (6,)
        sub_idx = ROUTE_SUBSETS[l2_pred_name]["idx"]  # list of two indices
        local = q3[sub_idx]
        k_local = int(np.argmax(local))
        k3 = sub_idx[k_local]               # index in 0..5
        l3_pred_name = SUBCLASS_LABELS[k3]
        l3_conf = float(local[k_local])
        l3_cam = compute_gradcam(l3, x, l3_t, target_index=k3)
        l3_overlay = overlay_cam(rgb, l3_cam, alpha=cam_alpha)
        # For bar chart, show only the two relevant subclass probs
        q3_vis = pd.Series({SUBCLASS_LABELS[i]: float(q3[i]) for i in sub_idx}).sort_values(ascending=False)

# ------------ Results ------------
st.markdown("### 🩺 Model Decision")
if l1_pred == "Normal":
    st.success(f"**L1:** Normal  (p_normal={p_normal:.2f} ≥ τ={tau:.2f})")
else:
    st.warning(f"**L1:** Abnormal (p_normal={p_normal:.2f} < τ={tau:.2f})")
    st.success(f"**L2 Diagnosis:** {l2_pred_name}  (conf={l2_conf:.2f})")
    if l3_pred_name is not None:
        st.info(f"**L3 Subclass:** {l3_pred_name}  (conf={l3_conf:.2f})")
    elif l2_pred_name in ROUTE_SUBSETS and l3 is None:
        st.error("Subclass model not loaded. Add `runs/SUBCLASS/subclass/best.pt` to enable L3.")
    else:
        st.caption("Subclass step not applicable (only for Eczema / Fungal / Pox).")

st.markdown("### 🔎 Grad-CAM Overlays")
cols = st.columns(3)
cols[0].image(l1_overlay, caption=f"L1 (ResNet-50): {l1_pred}", use_container_width=True)
if l2_overlay is not None:
    cols[1].image(l2_overlay, caption=f"L2 (DenseNet-121): {l2_pred_name}", use_container_width=True)
if l3_overlay is not None:
    cols[2].image(l3_overlay, caption=f"L3 (DenseNet-121 Subclass): {l3_pred_name}", use_container_width=True)

# Probability charts
if q2 is not None:
    st.markdown("### 📊 L2 Class Probabilities")
    st.bar_chart(pd.Series({L2_LABELS[i]: float(q2[i]) for i in range(len(L2_LABELS))})
                 .sort_values(ascending=False))
if q3_vis is not None:
    st.markdown("### 🧪 L3 Subclass Probabilities")
    st.bar_chart(q3_vis)
