import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

st.set_page_config(page_title="Traffic Sign Recognizer", page_icon="🚦")


## Model definition (must match the notebook exactly so the saved weights load)
class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


## Load the trained model once and cache it
@st.cache_resource
def load_model(path="traffic_sign_cnn.pth"):
    ckpt = torch.load(path, map_location="cpu")
    model = TrafficSignCNN(ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["class_names"], ckpt["img_size"]


## Same preprocessing as training: resize, scale to 0-1, normalize with mean/std 0.5
def preprocess(img, img_size):
    arr = np.array(img.convert("RGB").resize((img_size, img_size)), dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


## Page layout
st.title("🚦 Traffic Sign Recognizer")
st.write("Upload a photo of a traffic sign. A small CNN trained on the GTSRB dataset (43 German sign types) will name it.")

try:
    model, class_names, img_size = load_model()
except FileNotFoundError:
    st.error("traffic_sign_cnn.pth not found. Run the notebook (Step 23) and keep that file in the same folder as app.py.")
    st.stop()

threshold = st.sidebar.slider("Minimum confidence to trust a prediction", 0.0, 1.0, 0.7, 0.05)
st.sidebar.caption("Below this value the app says 'Not sure' instead of guessing.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded)
    col1, col2 = st.columns(2)
    col1.image(img, caption="Your image", use_container_width=True)

    with torch.no_grad():
        probs = torch.softmax(model(preprocess(img, img_size)), dim=1)[0]
    top_probs, top_ids = probs.topk(3)
    best_name = class_names[top_ids[0].item()]
    best_prob = top_probs[0].item()

    with col2:
        if best_prob >= threshold:
            st.success(f"**{best_name}**")
        else:
            st.warning(f"**Not sure** (best guess: {best_name})")
        st.metric("Confidence", f"{best_prob * 100:.1f}%")
        st.write("Top 3 guesses")
        st.bar_chart({class_names[i.item()]: p.item() for i, p in zip(top_ids, top_probs)}, horizontal=True)

st.divider()
st.caption(
    "Limits: the model only knows 43 German sign types and expects a cropped photo where the sign fills most of the frame. "
    "Signs outside those classes (for example, Indian-specific signs) will still get a guess, sometimes with high confidence, "
    "because the model has no 'unknown' option."
)
