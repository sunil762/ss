"""
Diabetic Retinopathy Detection — Streamlit App
================================================

Combined frontend + backend: this single app serves the UI, runs inference,
and logs every prediction to a CSV for the Power BI dashboard.

Run locally:
    streamlit run app.py

Deploy live (free):
    1. Push this project to a public GitHub repo
    2. Go to https://share.streamlit.io , sign in with GitHub
    3. "New app" -> pick the repo -> main file path: app/app.py -> Deploy
    4. You get a public URL like https://your-app.streamlit.app

NOTE: this app works in "demo mode" with a random-weight model if
best_model.pth is not present, so you can test the UI before training is
done. Replace it with your trained weights (from model_training/train.py)
before treating any output as meaningful.
"""

import os
import io
import csv
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
CLASS_COLORS = ["#2E7D32", "#9CCC65", "#F9A825", "#EF6C00", "#C62828"]
NUM_CLASSES = 5
IMG_SIZE = 224
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")
LOG_PATH = os.path.join(os.path.dirname(__file__), "predictions_log.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Diabetic Retinopathy Detection", page_icon="🩺", layout="wide")

# ---------------------------------------------------------------------------
# MODEL LOADING (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    demo_mode = not os.path.exists(MODEL_PATH)
    if not demo_mode:
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model, demo_mode


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict(model, image: Image.Image):
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs, tensor


def grad_cam(model, tensor, pred_idx):
    """Lightweight Grad-CAM against the last conv block of EfficientNet-B0."""
    activations, gradients = [], []

    target_layer = model.features[-1]

    def fwd_hook(_, __, output):
        activations.append(output)

    def bwd_hook(_, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    output = model(tensor)
    output[0, pred_idx].backward()

    h1.remove(); h2.remove()

    act = activations[0].detach()[0]      # (C, H, W)
    grad = gradients[0].detach()[0]       # (C, H, W)
    weights = grad.mean(dim=(1, 2))       # (C,)
    cam = torch.relu((weights[:, None, None] * act).sum(0))
    cam = cam / (cam.max() + 1e-8)
    return cam.cpu().numpy()


def overlay_heatmap(image: Image.Image, cam: np.ndarray):
    import matplotlib.cm as cm
    image = image.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    heat = Image.fromarray(np.uint8(cm.jet(cam)[:, :, :3] * 255)).resize((IMG_SIZE, IMG_SIZE))
    blended = Image.blend(image, heat, alpha=0.4)
    return blended


def log_prediction(filename, pred_idx, probs):
    row = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "filename": filename,
        "predicted_class": pred_idx,
        "predicted_label": CLASS_NAMES[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "prob_no_dr": round(float(probs[0]), 4),
        "prob_mild": round(float(probs[1]), 4),
        "prob_moderate": round(float(probs[2]), 4),
        "prob_severe": round(float(probs[3]), 4),
        "prob_proliferative": round(float(probs[4]), 4),
    }
    file_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return row


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🩺 Diabetic Retinopathy Detection")
st.caption(
    "Upload a retinal fundus photo to classify diabetic retinopathy severity "
    "using a CNN (EfficientNet-B0, transfer learning)."
)
st.warning(
    "**Not a diagnostic tool.** This is an educational/portfolio project. "
    "It has not been clinically validated and must not be used to make real "
    "medical decisions. Always consult an ophthalmologist for diagnosis.",
    icon="⚠️",
)

model, demo_mode = load_model()
if demo_mode:
    st.info(
        "Running in **demo mode** — no trained weights found (`best_model.pth`). "
        "Predictions right now come from an untrained network and are meaningless. "
        "Train the model with `model_training/train.py` and drop `best_model.pth` "
        "into the `app/` folder to get real predictions.",
        icon="ℹ️",
    )

left, right = st.columns([1, 1])

with left:
    uploaded = st.file_uploader("Upload a fundus image", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

with right:
    if uploaded:
        pred_idx, probs, tensor = predict(model, image)
        label = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx]

        st.subheader(f"Prediction: {label}")
        st.metric("Confidence", f"{confidence*100:.1f}%")

        prob_df = pd.DataFrame({"Severity": CLASS_NAMES, "Probability": probs})
        st.bar_chart(prob_df.set_index("Severity"))

        with st.expander("Show Grad-CAM explanation (which regions drove the prediction)"):
            try:
                cam = grad_cam(model, tensor, pred_idx)
                overlay = overlay_heatmap(image, cam)
                st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
            except Exception as e:
                st.caption(f"Grad-CAM unavailable: {e}")

        log_prediction(uploaded.name, pred_idx, probs)
        st.caption("This prediction was logged to `predictions_log.csv` for the analytics dashboard.")
    else:
        st.info("Upload an image on the left to see a prediction here.")

st.divider()
with st.expander("Recent predictions (session log)"):
    if os.path.exists(LOG_PATH):
        log_df = pd.read_csv(LOG_PATH).tail(20).iloc[::-1]
        st.dataframe(log_df, use_container_width=True)
        st.download_button(
            "Download full predictions_log.csv",
            data=open(LOG_PATH, "rb").read(),
            file_name="predictions_log.csv",
            mime="text/csv",
        )
    else:
        st.caption("No predictions logged yet.")
